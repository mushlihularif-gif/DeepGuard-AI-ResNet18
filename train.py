"""Reproducible ResNet18 training; never runs during Flask startup.

Train selects weights; validation selects epoch; calibration fits temperature;
test is evaluated once after selection. No private Downloads-folder training.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from modeling import CLASS_TO_IDX, PREPROCESS, make_model, preprocessing, rgb_image
from prepare_data import sources


class Images(Dataset):
    def __init__(self, datasets, rows, training=False):
        self.datasets, self.rows, self.transform = datasets, rows, preprocessing(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = rgb_image(self.datasets[row["source"]][row["index"]]["image"])
        return self.transform(image), row["label"]


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return torch.cat([model(images.to(device)).cpu() for images, _ in loader])


def metrics(logits, rows, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=1)[:, 1].numpy()
    labels = np.array([r["label"] for r in rows])
    result = {}
    for domain in ["all"] + sorted({r["domain"] for r in rows}):
        mask = np.ones(len(rows), dtype=bool) if domain == "all" else np.array([r["domain"] == domain for r in rows])
        y, p = labels[mask], probs[mask]
        pred = p >= 0.5
        tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
        fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
        decided = (p <= 0.2) | (p >= 0.8)
        result[domain] = {"n": len(y), "accuracy": float((pred == y).mean()),
                         "balanced_accuracy": 0.5 * (tn / max(tn + fp, 1) + tp / max(tp + fn, 1)),
                         "ai_recall": tp / max(tp + fn, 1), "real_recall": tn / max(tn + fp, 1),
                         "confusion_matrix_real_fake": [[tn, fp], [fn, tp]],
                         "brier": float(((p - y) ** 2).mean()),
                         "decision_coverage": float(decided.mean()),
                         "decided_accuracy": float((pred[decided] == y[decided]).mean()) if decided.any() else None}
    return result


def main(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    datasets = sources(args.tiny_train, args.tiny_val, args.modern_cache)
    if {k: v._fingerprint for k, v in datasets.items()} != manifest["source_fingerprints"]:
        raise ValueError("Dataset changed since the manifest was created")
    loaders = {name: DataLoader(Images(datasets, rows, name == "train"), batch_size=args.batch_size,
                               shuffle=name == "train", num_workers=args.workers,
                               pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
               for name, rows in manifest["splits"].items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = make_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 5e-5},
        {"params": model.fc.parameters(), "lr": 5e-4}], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history = -1, []
    print("Training", device, {k: len(v.dataset) for k, v in loaders.items()}, flush=True)
    for epoch in range(args.epochs):
        model.train()
        started, loss_total, correct, total = time.time(), 0.0, 0, 0
        for step, (images, labels) in enumerate(loaders["train"], 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
            if step % 25 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} step {step}/{len(loaders['train'])} loss={loss_total/total:.4f} acc={correct/total:.4f} seconds={time.time()-started:.0f}", flush=True)
        scheduler.step()
        report = metrics(predict(model, loaders["validation"], device), manifest["splits"]["validation"])
        score = report["all"]["balanced_accuracy"]
        history.append({"epoch": epoch + 1, "validation": report, "seconds": time.time() - started})
        print("VALIDATION", epoch + 1, json.dumps(report), flush=True)
        if score > best:
            best = score
            torch.save({"schema_version": 1, "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "class_to_idx": CLASS_TO_IDX, "preprocess": PREPROCESS, "temperature": 1.0,
                        "epoch": epoch + 1, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                        "training": {"seed": args.seed, "initialization": "ImageNet1K_V1", "n": len(loaders["train"].dataset)},
                        "limitations": manifest["limitations"]}, output)
    checkpoint = torch.load(output, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    calibration_logits = predict(model, loaders["calibration"], device).clone()
    calibration_labels = torch.tensor([r["label"] for r in manifest["splits"]["calibration"]])
    # Scalar temperature cannot change class order. Fit only on calibration split.
    candidates = torch.logspace(-1, 1, 300)
    losses = [torch.nn.functional.cross_entropy(calibration_logits / t, calibration_labels).item() for t in candidates]
    temperature = float(candidates[int(np.argmin(losses))])
    checkpoint["temperature"] = temperature
    checkpoint["decision_policy"] = {"real_max": 0.2, "fake_min": 0.8,
                                     "note": "Conservative fixed review band; no guaranteed error rate"}
    test_logits = predict(model, loaders["test"], device)
    report = {"manifest_sha256": checkpoint["manifest_sha256"], "selected_epoch": checkpoint["epoch"],
              "temperature": temperature, "history": history, "limitations": manifest["limitations"],
              "test": metrics(test_logits, manifest["splits"]["test"], temperature), "baselines": {}}
    for path in args.baseline:
        baseline = make_model().to(device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        baseline.load_state_dict(state.get("state_dict", state), strict=True)
        report["baselines"][Path(path).stem] = metrics(predict(baseline, loaders["test"], device), manifest["splits"]["test"])
        del baseline
    checkpoint["evaluation"] = report["test"]
    torch.save(checkpoint, output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL", json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny-train", required=True)
    p.add_argument("--tiny-val", required=True)
    p.add_argument("--modern-cache", required=True)
    p.add_argument("--manifest", default="work/manifest.json")
    p.add_argument("--output", default="models/candidate.pt")
    p.add_argument("--report", default="reports/evaluation.json")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--baseline", action="append", default=[])
    main(p.parse_args())
