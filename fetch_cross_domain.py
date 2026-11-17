"""Download only the 84 public pilot images and verify their recorded hashes.

Explicit command; never invoked by the app. Does not fetch the full dataset.
"""
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from PIL import Image
import io


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / "reports/cross_domain.json").read_text())
    plan = json.loads((root / "reports/cross_domain_plan.json").read_text())
    destination = root / "work/external_eval"
    destination.mkdir(parents=True, exist_ok=True)
    expected = {row["row_idx"]: row for row in report["samples"]}
    offsets = [offset for offset in plan["offsets"] if offset in expected]

    def fetch(offset):
        params = {"dataset": plan["dataset"], "config": "default", "split": plan["split"],
                  "offset": offset, "length": plan["rows_per_request"]}
        with urlopen("https://datasets-server.huggingface.co/rows?" + urlencode(params), timeout=60) as response:
            return json.load(response)["rows"]

    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entries in pool.map(fetch, offsets):
            for entry in entries:
                index = entry["row_idx"]
                if index not in expected:
                    continue
                record, row = expected[index], entry["row"]
                raw = base64.b64decode(row["image_data"])
                if hashlib.sha256(raw).hexdigest() != record["sha256"] or int(row["label"]) != record["label"]:
                    raise ValueError(f"Dataset row {index} changed; refusing to replace the recorded pilot")
                with Image.open(io.BytesIO(raw)) as image:
                    suffix = ".jpg" if image.format == "JPEG" else ".png"
                path = destination / (str(index) + suffix)
                path.write_bytes(raw)
                manifest.append({k: record[k] for k in ("row_idx", "label", "generator", "real_source", "sha256", "content_sha256_64rgb")})
                manifest[-1]["path"] = str(path)
    if len(manifest) != len(expected):
        raise ValueError("Incomplete pilot; no manifest published")
    (destination / "manifest.json").write_text(json.dumps(sorted(manifest, key=lambda r: r["row_idx"]), indent=2))
    print(f"Verified {len(manifest)} images: {destination / 'manifest.json'}")


if __name__ == "__main__":
    main()
