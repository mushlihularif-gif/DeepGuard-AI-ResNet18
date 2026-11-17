"""Evaluate labelled images through the browser API.

CSV columns: path,label,group. label: REAL or FAKE; group: camera,generated,
ai_edited. Relative paths resolve against the CSV directory. Outputs aggregates.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from app import app, model_status


def evaluate(manifest):
    client, groups = app.test_client(), defaultdict(Counter)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Evaluation CSV is empty")
    for row in rows:
        expected = row["label"].upper()
        if expected not in {"REAL", "FAKE"}:
            raise ValueError("Ground truth must be explicitly REAL or FAKE")
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        group = row.get("group") or "unspecified"
        with path.open("rb") as stream:
            response = client.post("/api/predict", data={"image": (stream, path.name)})
        result = response.get_json()
        for key in {"all", group}:
            counts = groups[key]
            counts["total"] += 1
            if response.status_code != 200:
                counts["errors"] += 1
            elif result["label"] == "UNCERTAIN":
                counts["uncertain"] += 1
            else:
                predicted = "FAKE" if result["label"] == "AI_DISCLOSED" else result["label"]
                counts["decided"] += 1
                counts["correct" if predicted == expected else "wrong"] += 1
                counts[f"expected_{expected}_predicted_{predicted}"] += 1
                if result.get("decision_basis") == "verified_c2pa":
                    counts["c2pa_decisions"] += 1
    return {"model_sha256": model_status()["model_sha256"], "groups": {
        name: {**counts, "coverage": counts["decided"] / counts["total"],
               "decided_accuracy": counts["correct"] / counts["decided"] if counts["decided"] else None}
        for name, counts in groups.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("work/upload_evaluation.json"))
    args = parser.parse_args()
    report = evaluate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
