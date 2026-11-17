# DeepGuard AI — ResNet18

Proyek **Artificial Intelligence for Security**, Teknik Informatika, Kelompok 8.
Aplikasi Flask untuk melihat indikasi citra AI menggunakan ResNet18.

**Hasil adalah indikasi, bukan sertifikat keaslian.** Model bisa salah pada foto
kamera, generator yang belum dikenal, screenshot, kompresi, ilustrasi, dan edit
sebagian gambar. Mengganti baju/wajah/latar dengan AI tetap termasuk penggunaan AI,
tetapi kemampuan mendeteksi kasus tersebut **belum tervalidasi** pada model ini.
Skor 90% bukan berarti 90% area gambar diedit.

## Menjalankan

Python 3.10–3.12; CPU didukung, CUDA opsional.

```bash
git clone https://github.com/mushlihularif-gif/DeepGuard-AI-ResNet18.git
cd DeepGuard-AI-ResNet18
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Buka [Deteksi Citra](http://127.0.0.1:5000/deteksi). Dataset tidak diperlukan
untuk menjalankan aplikasi: cukup checkpoint sekitar 45 MB yang disertakan.
Startup tidak mengunduh data, tidak melatih ulang, dan tidak mengubah bobot.
Restart Flask setelah mengganti checkpoint.

## Perbaikan November 2026

- Pemetaan label eksplisit, termasuk `RealArt → REAL` dan `AiArtData → FAKE`.
  Versi V3 sebelumnya memetakan dua kelas Hemg sebagai FAKE, dan mengambil 20.000
  baris awal yang seluruhnya AI.
- Pelatihan baru dari ImageNet dengan data seimbang; tanpa mengambil foto pribadi
  dari folder Downloads dan tanpa label yang ditebak dari nama file.
- Split train, pemilihan epoch, kalibrasi, dan test terpisah. Manifest indeks/hash
  disertakan, termasuk keterbatasan pemeriksaan duplikat.
- Nama file, EXIF kamera, dimensi khas AI, warna, dan sudut terang tidak lagi
  menjadi aturan tambahan untuk mengubah skor.
- Checkpoint menyimpan kelas, preprocessing, temperatur, dan hasil evaluasi.
  Checkpoint lama tanpa kontrak metadata ditolak; tidak ada fallback bobot acak.
- Hasil `REAL`, `FAKE`, atau `UNCERTAIN`; klaim pemeriksaan FFT/noise sensor
  yang tidak benar-benar dilakukan telah dihapus. Tombol demo hasil palsu dihapus.

## Hasil dan batas benchmark

Hasil lengkap ada di [reports/evaluation.json](reports/evaluation.json) dan
[laporan audit](reports/AUDIT.md). Data: 12.000 latihan, 800 validasi, 800
kalibrasi, 1.600 uji, dengan kelas dan sumber seimbang. Metrik pada halaman web
dibaca dari checkpoint aktif, bukan angka yang ditulis tetap di HTML.

Sumber:
[Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) dan
[Hemg](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets).
Hemg mencakup karya seni non-AI, bukan hanya foto kamera. Tiny validation pernah
dipakai model lama, sehingga perbandingan model lama di subset Tiny tidak merupakan
uji independen model lama. Hash RGB 64×64 tidak menjamin semua near-duplicate hilang.
Belum ada benchmark independen untuk edit lokal atau generator terbaru.

## API

- `GET /api/model-status`: status, SHA-256 model, perangkat, dan metrik checkpoint.
- `POST /api/predict`: multipart field `image`, JPG/PNG, maksimum 10 MB dan
  25 megapiksel. Gambar diproses di memori, tidak disimpan oleh aplikasi.
- `prob_fake`: softmax setelah temperature scaling pada split kalibrasi.
  Skor ≤0,2 menghasilkan REAL; ≥0,8 FAKE; di antaranya UNCERTAIN.
  Ambang ini merupakan aturan peninjauan konservatif, bukan jaminan tingkat error.
- REAL berarti cenderung kelas non-AI, **tidak membuktikan gambar belum diedit AI**.
- Berkas tidak valid: HTTP 400; terlalu besar: 413; model tidak siap: 503.

## Pengujian dan training

```bash
python -m unittest discover -s tests -v
node tests/test_frontend.cjs
python evaluate_uploads.py data/ground_truth.csv
```

CSV pengujian eksternal memiliki kolom `path,label,group`; label harus diketahui
dari asal gambar, bukan tampilannya. Kelompok `camera`, `generated`, dan
`ai_edited` dilaporkan terpisah. Jangan masukkan gambar uji ke latihan.

[Petunjuk training lokal dan cloud](docs/TRAINING.md) mencakup alur Colab dan
streaming agar dataset besar tidak perlu disimpan di laptop. Skrip lama
`train_cuda.py`, `train_hybrid.py`, dan `train_v3_universal.py` kini hanya
meneruskan ke trainer baru dengan argumen eksplisit. Anchor dan flag kalibrasi lama
disimpan sebagai artefak historis, tidak dibaca aplikasi.

Lingkungan pengujian perbaikan: Python 3.12, torch 2.6.0+cu124, torchvision
0.21.0+cu124, Flask 3.1.3, Pillow 12.2.0, datasets 5.2.0, numpy 2.4.6.
