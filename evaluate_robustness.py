"""Evaluate a fixed resize/JPEG perturbation on the same held-out test images.

This is a robustness check, not another independent dataset or WhatsApp benchmark.
"""
import argparse
import io
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader
from modeling import load_checkpoint, preprocessing, rgb_image
from prepare_data import sources
from train import Images, metrics, predict


class SharedImage(Images):
    def __getitem__(self, index):
        row = self.rows[index]
        image = rgb_image(self.datasets[row["source"]][row["index"]]["image"])
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=70)
        stream.seek(0)
        with Image.open(stream) as decoded:
            return self.transform(decoded.convert("RGB")), row["label"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-train", required=True)
    parser.add_argument("--tiny-val", required=True)
    parser.add_argument("--modern-cache", required=True)
    parser.add_argument("--manifest", default="reports/data_manifest.json")
    parser.add_argument("--model", default="models/resnet18_model.pt")
    parser.add_argument("--output", default="reports/robustness.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    datasets = sources(args.tiny_train, args.tiny_val, args.modern_cache)
    if {k: v._fingerprint for k, v in datasets.items()} != manifest["source_fingerprints"]:
        raise ValueError("Dataset fingerprint mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.model, device)
    rows = manifest["splits"]["test"]
    loader = DataLoader(SharedImage(datasets, rows), batch_size=64, shuffle=False)
    report = {"transformation": "longest_side_max512_then_jpeg_quality70",
              "note": "Same test images, not independent additional samples; no retuning after evaluation",
              "metrics": metrics(predict(model, loader, device), rows, checkpoint["temperature"])}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
