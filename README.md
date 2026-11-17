# DeepGuard AI — Community Forensics + baseline ResNet18

Proyek **Artificial Intelligence for Security**, Teknik Informatika, Kelompok 8.
Aplikasi Flask untuk melihat indikasi citra AI. Mesin utama sekarang **Community
Forensics ViT-S/16**, bukan ResNet18. ResNet18 hasil latihan proyek tetap tersedia
sebagai baseline akademik. Perubahan arsitektur ini harus disebutkan dalam laporan
kuliah; hasil ViT tidak boleh diklaim sebagai hasil ResNet18.

**Hasil adalah indikasi, bukan sertifikat keaslian.** Model bisa salah pada foto
kamera, generator yang belum dikenal, screenshot, kompresi, ilustrasi, dan edit
sebagian gambar. Mengganti baju/wajah/latar dengan AI tetap termasuk penggunaan AI,
tetapi kemampuan mendeteksi kasus tersebut **belum tervalidasi** pada model ini.
Skor 90% bukan berarti 90% area gambar diedit.

## Content Credentials: bukti asal dipisahkan dari prediksi visual

Unggahan sekarang juga diperiksa dengan SDK C2PA. Jika **manifest aktif** mencatat
`trainedAlgorithmicMedia` atau `compositeWithTrainedAlgorithmicMedia`, tanda tangan
valid, hash cocok dengan berkas, dan sertifikat dipercaya oleh snapshot daftar
resmi C2PA, hasil utama menjadi **Penggunaan AI tercatat** (`AI_DISCLOSED`).
Skor visual tetap ditampilkan tanpa diubah: model dapat salah walaupun bukti asal
mencatat AI. Pemeriksaan ini tidak melihat nama file atau menambahkan aturan EXIF.

Infografik yang dilaporkan pengguna dan ilustrasi mesin pesawat sama-sama memiliki
kredensial AI valid dari OpenAI Media Service. Keduanya kini dikenali lewat bukti
asal tersebut, bukan lewat peningkatan kemampuan model visual. Setelah metadata
dihilangkan, bukti ini hilang dan kelemahan model visual tetap ada. Tidak adanya
C2PA, kredensial rusak, atau kredensial kamera **tidak membuktikan bebas AI**.

Pemeriksaan tertanam berjalan di memori, tanpa mengirim unggahan atau mengambil
manifest dari internet. Kepercayaan memakai snapshot lokal; pencabutan sertifikat
terkini dan kepercayaan otoritas timestamp tidak diverifikasi online. Lihat
[kebijakan dan pengujian C2PA](docs/PROVENANCE.md).

## Menjalankan

Python 3.10–3.12; CPU didukung, CUDA opsional.

```bash
git clone https://github.com/mushlihularif-gif/DeepGuard-AI-ResNet18.git
cd DeepGuard-AI-ResNet18
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-detector.txt
python setup_detector.py
python app.py
```

Buka [Deteksi Citra](http://127.0.0.1:5000/deteksi). Dataset tidak diperlukan
untuk menjalankan aplikasi. Setup mengunduh bobot Community Forensics sekitar
87 MB sekali dari sumber resmi, memeriksa SHA-256, dan mendukung melanjutkan
unduhan yang terputus. Dataset ratusan GB tidak diunduh. Bobot ini tidak masuk Git;
clone baru perlu menjalankan setup. Model ResNet18 sekitar 45 MB tetap disertakan.
Startup tidak mengunduh data, tidak melatih ulang, dan tidak mengubah bobot.
Restart Flask setelah mengganti checkpoint. Inferensi berjalan lokal, bukan API
cloud: unggahan tidak dikirim ke Hugging Face. Cloud dapat digunakan untuk training
terpisah; tidak ada training cloud yang dijalankan oleh aplikasi ini.

Untuk baseline saja (PowerShell; cukup `requirements.txt`):

```powershell
$env:DEEPGUARD_ENGINE = 'resnet18'
python app.py
# Kembali ke mesin utama:
Remove-Item Env:DEEPGUARD_ENGINE
```

`DEEPGUARD_MODEL_PATH` opsional mengganti lokasi checkpoint sesuai mesin.
Model hilang/rusak menghasilkan HTTP 503, tanpa fallback diam-diam. Mesin utama
memakai resize sisi pendek 440 lalu center crop 384 sesuai preprocessing penulis;
edit di tepi gambar bisa terlewat. Skor sigmoid belum dikalibrasi pada data proyek.

## Uji lintas sumber: mengapa model diganti

Pada pilot **84 gambar** dari CommunityForensics-Eval / CompEval, berbeda dari
dua dataset latihan ResNet18 lokal:

| Uji biner pada ambang 0,5 | ResNet18 | Community Forensics |
| --- | ---: | ---: |
| Benar, gambar asli ukuran sumber | 48/84 (57,1%) | 77/84 (91,7%) |
| Non-AI salah dianggap AI | 11/32 | 0/32 |
| AI salah dianggap non-AI | 25/52 | 7/52 |
| Benar setelah resize maks. 512 + JPEG kualitas 70 | 48/84 (57,1%) | 67/84 (79,8%) |

**Ini pilot kecil, bukan klaim akurasi umum.** Sebanyak 11/32 permintaan data gagal;
84/128 target gambar berhasil diterima. Sampel tidak seimbang. Overlap hash dengan
split lokal sudah diperiksa, tetapi near-duplicate dan overlap dengan seluruh
korpus pralatih model eksternal belum diaudit. Pilot dipakai memilih mesin, jadi
konfirmasi selanjutnya memerlukan set uji baru yang belum dipakai memilih model.
Tidak ada fine-tuning, pengubahan label, atau penyesuaian ambang pada pilot.

Pada ambang tampilan aplikasi (REAL ≤0,2; FAKE ≥0,8), model utama memberikan
74 keputusan benar, 4 salah, dan 6 ragu pada 84 gambar sumber. Setelah kompresi:
61 benar, 12 salah, 11 ragu. Jangan menyamakan akurasi biner dengan keputusan UI.
Gambar ilustrasi mesin pesawat yang dilaporkan pengguna tetap **UNCERTAIN**
(skor sekitar 39,9%); kasus itu tidak dipakai melatih atau memilih ambang.
Deteksi perubahan baju/wajah/latar **belum terselesaikan atau tervalidasi**.

Lihat [laporan pilot](reports/CROSS_DOMAIN.md), [hasil per gambar](reports/cross_domain.json),
[uji kompresi](reports/cross_domain_jpeg.json), dan [protokol](reports/cross_domain_plan.json).
Halaman web hanya menampilkan metrik pilot yang cocok dengan hash checkpoint aktif.

Sumber model: [Community Forensics, CVPR 2025](https://github.com/JeongsooP/Community-Forensics),
[checkpoint resmi](https://huggingface.co/OwensLab/commfor-model-384).
Model dan kode sumber upstream berlisensi MIT; lihat [pemberitahuan lisensi](docs/COMMUNITY_FORENSICS_LICENSE.txt).

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

## Benchmark latihan ResNet18 sebelumnya

Hasil lengkap ada di [reports/evaluation.json](reports/evaluation.json) dan
[laporan audit](reports/AUDIT.md). Data: 12.000 latihan, 800 validasi, 800
kalibrasi, 1.600 uji, dengan kelas dan sumber seimbang. Angka 86,2% dari subset ini
tidak mewakili gambar internet; pilot lintas sumber di atas menunjukkan penurunan
ke 57,1%. Benchmark lama dipertahankan sebagai catatan penelitian.

Sumber:
[Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) dan
[Hemg](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets).
Hemg mencakup karya seni non-AI, bukan hanya foto kamera. Tiny validation pernah
dipakai model lama, sehingga perbandingan model lama di subset Tiny tidak merupakan
uji independen model lama. Hash RGB 64×64 tidak menjamin semua near-duplicate hilang.
Belum ada benchmark independen untuk edit lokal.

## API

- `GET /api/model-status`: mesin, arsitektur, ukuran input, jenis skor, SHA-256,
  perangkat, dan hasil pilot yang sesuai checkpoint.
- `POST /api/predict`: multipart field `image`, JPG/PNG, maksimum 10 MB dan
  25 megapiksel. Gambar diproses di memori, tidak disimpan oleh aplikasi.
- `prob_fake`: skor AI. Community Forensics: sigmoid mentah (`uncalibrated_sigmoid`);
  ResNet18: softmax setelah temperature scaling (`temperature_scaled_softmax`).
  Nama field dipertahankan untuk kompatibilitas, bukan jaminan probabilitas benar.
  Skor ≤0,2 menghasilkan REAL; ≥0,8 FAKE; di antaranya UNCERTAIN.
  Ambang ini merupakan aturan peninjauan konservatif, bukan jaminan tingkat error.
- `model_label`: keputusan model visual sebelum pemeriksaan bukti asal.
- `decision_basis`: `visual_model` atau `verified_c2pa`. `label=AI_DISCLOSED`
  hanya diberikan untuk deklarasi AI aktif yang lolos validasi C2PA dan trust.
  `provenance` berisi status, ringkasan, serta nama penandatangan bila dipercaya.
  Tidak ada pengubahan `prob_fake` menjadi 100% untuk menyesuaikan hasil C2PA.
- REAL berarti cenderung kelas non-AI, **tidak membuktikan gambar belum diedit AI**.
- Berkas tidak valid: HTTP 400; terlalu besar: 413; model tidak siap: 503.

## Pengujian dan training

```bash
python -m unittest discover -s tests -v
node tests/test_frontend.cjs
python evaluate_uploads.py data/ground_truth.csv
# Opsional, mengunduh hanya 84 gambar publik pilot:
python fetch_cross_domain.py
python evaluate_cross_domain.py work/external_eval/manifest.json --community-model models/community_forensics.safetensors
python evaluate_cross_domain.py work/external_eval/manifest.json --community-model models/community_forensics.safetensors --jpeg-share --output work/cross_domain_jpeg.json
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
