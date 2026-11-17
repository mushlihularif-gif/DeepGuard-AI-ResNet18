"""Compare fixed models on a labelled external manifest, without training/tuning.

Manifest fields: path, label (0=non-AI, 1=AI), generator, row_idx, sha256.
Images remain local. The report excludes local filenames and image contents.
Community Forensics is optional; install requirements-detector.txt to compare it.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import torch
from PIL import Image
from modeling import load_checkpoint, preprocessing, rgb_image


def summarize(rows, key):
    matrix = [[0, 0], [0, 0]]
    decided = correct_decided = 0
    ui_matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        score, label = row[key], row["label"]
        prediction = int(score >= 0.5)
        matrix[label][prediction] += 1
        ui_matrix[label][0 if score <= 0.2 else 1 if score >= 0.8 else 2] += 1
        if score <= 0.2 or score >= 0.8:
            decided += 1
            correct_decided += prediction == label
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {"n": len(rows), "confusion_matrix_real_fake": matrix,
            "ui_confusion_real_fake_uncertain": ui_matrix,
            "accuracy": (tn + tp) / len(rows) if rows else None,
            "balanced_accuracy": (tn/(tn+fp) + tp/(tp+fn))/2 if (tn+fp)*(tp+fn) else None,
            "real_false_positive_rate": fp/(tn+fp) if tn+fp else None,
            "ai_false_negative_rate": fn/(tp+fn) if tp+fn else None,
            "decision_coverage": decided/len(rows) if rows else None,
            "decided_accuracy": correct_decided/decided if decided else None}


def evaluate(manifest, community_path=None, jpeg_share=False):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, metadata = load_checkpoint("models/resnet18_model.pt", device)
    engines = {"resnet18": (baseline, preprocessing(), metadata["temperature"])}
    hashes = {"resnet18": hashlib.sha256(Path("models/resnet18_model.pt").read_bytes()).hexdigest()}
    if community_path:
        from community_detector import load_community, SHA256
        model, transform, meta = load_community(community_path, device)
        engines["community_forensics"] = model, transform, meta["temperature"]
        hashes["community_forensics"] = SHA256
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("Empty evaluation manifest")
    results = []
    with torch.inference_mode():
        for row in rows:
            if row["label"] not in (0, 1):
                raise ValueError("Ground-truth labels must be 0 or 1")
            path = Path(row["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Evaluation image hash mismatch")
            result = {k: v for k, v in row.items() if k != "path"}
            with Image.open(path) as image:
                image = rgb_image(image)
                if jpeg_share:
                    image.thumbnail((512, 512))
                    buffer = io.BytesIO()
                    image.save(buffer, "JPEG", quality=70)
                    buffer.seek(0)
                    with Image.open(buffer) as decoded:
                        image = decoded.convert("RGB")
                for name, (model, transform, temperature) in engines.items():
                    result[name] = float(torch.softmax(model(transform(image).unsqueeze(0).to(device))/temperature, dim=1)[0, 1])
            results.append(result)
    return {"model_sha256": hashes,
            "input_transform": "long_side_max512_jpeg70" if jpeg_share else "original",
            "overall": {name: summarize(results, name) for name in engines},
            "per_generator": {group: {name: summarize([r for r in results if r["generator"] == group], name)
                                      for name in engines} for group in sorted({r["generator"] for r in results})},
            "samples": results,
            "limitations": ["Small external pilot; not a universal accuracy estimate.",
                            "No tuning/training performed on these images.",
                            "Source labels are inherited from the dataset authors.",
                            "Exact normalized overlap with local training splits was excluded during collection; near-duplicate/pretrained-corpus overlap is not fully audited.",
                            "Partial AI edits remain unvalidated."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--community-model")
    parser.add_argument("--output", default="work/cross_domain.json")
    parser.add_argument("--jpeg-share", action="store_true", help="Also simulate a resize and JPEG re-encode before inference")
    args = parser.parse_args()
    report = evaluate(args.manifest, args.community_model, args.jpeg_share)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
