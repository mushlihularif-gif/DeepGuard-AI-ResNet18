"""Explicit, resumable download of the pinned detector weights (87.3 MB, no dataset)."""
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen
from community_detector import FILE_SIZE, REPOSITORY, REVISION, SHA256

DESTINATION = Path(__file__).resolve().parent / "models/community_forensics.safetensors"


def verified(path):
    return path.exists() and path.stat().st_size == FILE_SIZE and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256


def main():
    if verified(DESTINATION):
        print("Detector sudah tersedia; SHA-256 sesuai.")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    partial = DESTINATION.with_suffix(".partial")
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.safetensors"
    position = partial.stat().st_size if partial.exists() else 0
    if position > FILE_SIZE:
        raise ValueError(f"Partial file too large; remove {partial} and retry")
    # Bounded requests avoid restarting the complete file on a slow connection.
    while position < FILE_SIZE:
        end = min(position + 2 * 1024 * 1024, FILE_SIZE) - 1
        request = Request(url + f"?range_start={position}", headers={"Range": f"bytes={position}-{end}"})
        with urlopen(request, timeout=60) as response:
            expected_range = f"bytes {position}-{end}/{FILE_SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server did not honor the requested byte range; file was not changed")
            block = response.read(end - position + 2)
        if len(block) != end - position + 1:
            raise ValueError("Incomplete download block; rerun to resume")
        with partial.open("ab") as stream:
            stream.write(block)
        position += len(block)
        print(f"{position / FILE_SIZE:.0%}", flush=True)
    if not verified(partial):
        raise ValueError(f"SHA-256 mismatch; remove {partial} and retry")
    partial.replace(DESTINATION)
    print("Detector siap. Restart aplikasi untuk memuatnya.")


if __name__ == "__main__":
    main()
