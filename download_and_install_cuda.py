import os
import sys
import time
import urllib.request
import subprocess

sys.stdout.reconfigure(line_buffering=True)

DOWNLOAD_DIR = r"E:\cuda_wheels"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

FILES = [
    {
        "name": "torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl",
        "url": "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl"
    },
    {
        "name": "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl",
        "url": "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl"
    }
]

def download_with_resume(url, filepath):
    filename = os.path.basename(filepath)
    # Cek ukuran berkas awal
    initial_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0

    # Dapatkan total ukuran dari server
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_bytes = int(resp.headers.get("Content-Length", 0))

    if initial_bytes >= total_bytes and total_bytes > 0:
        print(f"[OK] {filename} sudah selesai diunduh ({total_bytes / (1024*1024):.1f} MB).")
        return filepath

    print(f"\n[DOWNLOAD] Mengunduh {filename} ({total_bytes / (1024*1024):.1f} MB)...")
    if initial_bytes > 0:
        print(f"           Melanjutkan dari byte ke-{initial_bytes:,} ({initial_bytes / (1024*1024):.1f} MB)...")

    max_retries = 10
    retry = 0

    while retry < max_retries:
        try:
            curr_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            if curr_bytes >= total_bytes and total_bytes > 0:
                break

            headers = {}
            if curr_bytes > 0:
                headers["Range"] = f"bytes={curr_bytes}-"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp, open(filepath, "ab" if curr_bytes > 0 else "wb") as f:
                start_t = time.time()
                bytes_this_session = 0
                last_report = start_t

                while True:
                    chunk = resp.read(2 * 1024 * 1024)  # 2 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    curr_bytes += len(chunk)
                    bytes_this_session += len(chunk)

                    now = time.time()
                    if now - last_report >= 5.0 or curr_bytes >= total_bytes:
                        elapsed = now - start_t
                        speed_mb = (bytes_this_session / (1024 * 1024)) / max(elapsed, 0.1)
                        pct = (curr_bytes / total_bytes) * 100 if total_bytes > 0 else 0
                        remain_mb = (total_bytes - curr_bytes) / (1024 * 1024)
                        eta_s = remain_mb / max(speed_mb, 0.1)
                        print(f"  --> {curr_bytes / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.1f} MB/s | ETA: {eta_s:.0f}s")
                        last_report = now

            if os.path.getsize(filepath) >= total_bytes:
                print(f"[OK] Unduhan {filename} Selesai!")
                break

        except Exception as e:
            retry += 1
            print(f"[RETRY {retry}/{max_retries}] Koneksi terputus: {e}. Mencoba melanjutkan dalam 3 detik...")
            time.sleep(3)

    return filepath

# Download berkas
downloaded = []
for item in FILES:
    target = os.path.join(DOWNLOAD_DIR, item["name"])
    fp = download_with_resume(item["url"], target)
    downloaded.append(fp)

# Install wheels
print("\n[INSTALL] Memasang PyTorch + Torchvision CUDA ke environment...")
cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps"] + downloaded
ret = subprocess.run(cmd)

if ret.returncode == 0:
    print("\n[OK] Instalasi PyTorch CUDA Berhasil!")
    # Verifikasi
    ver_cmd = [sys.executable, "-c", "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"]
    subprocess.run(ver_cmd)
else:
    print("[ERROR] Gagal memasang wheel.")
