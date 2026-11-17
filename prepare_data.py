"""Build balanced splits without reading filenames as ground truth.

Stores only indices and content hashes, never copies private images.
The Hemg RealArt class is artwork/non-AI, not a verified camera-photo corpus.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import Counter
from datasets import Dataset, concatenate_datasets, load_from_disk
from modeling import label_mapping, rgb_image


def sources(tiny_train, tiny_val, modern_cache):
    if (Path(modern_cache) / "dataset_info.json").exists():
        return {"tiny_train": load_from_disk(tiny_train), "tiny_val": load_from_disk(tiny_val),
                "hemg": load_from_disk(modern_cache)}
    arrows = sorted(Path(modern_cache).rglob("*-train-*.arrow"))
    if not arrows:
        raise ValueError("No cached Hemg training Arrow shards found")
    # Require one cached revision; silently concatenating revisions duplicates data.
    if len({p.parent for p in arrows}) != 1:
        raise ValueError("Point --modern-cache at exactly one dataset revision")
    return {"tiny_train": load_from_disk(tiny_train), "tiny_val": load_from_disk(tiny_val),
            "hemg": concatenate_datasets([Dataset.from_file(str(p)) for p in arrows])}


def build(args):
    datasets = sources(args.tiny_train, args.tiny_val, args.modern_cache)
    rng = random.Random(args.seed)
    seen, splits, skipped = set(), {s: [] for s in ("test", "calibration", "validation", "train")}, Counter()
    pools = {}
    for name, ds in datasets.items():
        mapping = label_mapping(ds.features["label"].names)
        labels = ds.with_format("numpy", columns=["label"])[:]["label"]
        pools[name] = {c: [int(i) for i, raw in enumerate(labels) if mapping[int(raw)] == c] for c in (0, 1)}
        for pool in pools[name].values():
            rng.shuffle(pool)
        print(name, {c: len(v) for c, v in pools[name].items()}, mapping, flush=True)
    for split, amount in [("test", args.test_per_class), ("calibration", args.val_per_class),
                          ("validation", args.val_per_class), ("train", args.train_per_class)]:
        for domain, source in [("tiny", "tiny_train" if split == "train" else "tiny_val"), ("hemg", "hemg")]:
            for label in (0, 1):
                count = 0
                while count < amount:
                    if not pools[source][label]:
                        raise ValueError(f"Insufficient unique images: {source}/{split}/{label}")
                    index = pools[source][label].pop()
                    # Legacy V3 used the first 20,000 Hemg images. Keep them out of new evaluation.
                    if source == "hemg" and split != "train" and index < args.legacy_modern_prefix:
                        continue
                    try:
                        image = rgb_image(datasets[source][index]["image"])
                        image.load()
                        digest = hashlib.sha256(image.resize((64, 64)).tobytes()).hexdigest()
                        if digest in seen:
                            skipped["duplicate"] += 1
                            continue
                        seen.add(digest)
                        splits[split].append({"source": source, "index": index, "label": label,
                                              "domain": domain, "content_sha256_64rgb": digest})
                        count += 1
                    except (OSError, ValueError) as exc:
                        skipped["decode_error"] += 1
                print(split, domain, label, count, flush=True)
    result = {"seed": args.seed, "source_fingerprints": {k: v._fingerprint for k, v in datasets.items()},
              "class_to_idx": {"REAL": 0, "FAKE": 1}, "splits": splits, "skipped": dict(skipped),
              "limitations": ["Tiny validation was used by legacy models; not an independent legacy test.",
                              "Hemg RealArt includes artwork; not equivalent to camera authenticity.",
                              "64x64 RGB hash removes exact normalized duplicates, not all near duplicates.",
                              "No verified partial AI edits or modern generator holdout in this corpus."]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved", args.output, {k: len(v) for k, v in splits.items()}, dict(skipped), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny-train", required=True)
    p.add_argument("--tiny-val", required=True)
    p.add_argument("--modern-cache", required=True)
    p.add_argument("--output", default="work/manifest.json")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--train-per-class", type=int, default=3000)
    p.add_argument("--val-per-class", type=int, default=200)
    p.add_argument("--test-per-class", type=int, default=400)
    p.add_argument("--legacy-modern-prefix", type=int, default=20000,
                   help="Exclude this many legacy Hemg rows from evaluation; use 0 for newly streamed subsets")
    build(p.parse_args())
