# DeepGuard AI — Deteksi Citra Hasil Generative AI Menggunakan ResNet-18

Sistem deteksi dan analisis forensik citra digital berbasis Deep Learning (**ResNet-18**) dan antarmuka web (**Python Flask**) untuk membedakan antara **Foto Kamera Asli (Real)** dan **Citra Sintetis Generative AI (Fake)**.

Proyek ini dikembangkan sebagai pemenuhan tugas mata kuliah **Artificial Intelligence for Security** (Semester 5, Teknik Informatika) oleh **Kelompok 8**.

---

## Fitur Utama

- **Klasifikasi Citra Biner**: Membedakan foto kamera fisik asli vs gambar sintetis buatan AI (Stable Diffusion, Midjourney, DALL-E 3/ChatGPT, GANs).
- **Distribusi Probabilitas Softmax**: Menyajikan transparansi skor keyakinan persentase keaslian citra.
- **Antarmuka Web Modern**: Desain responsif berbasis HTML5, CSS3, dan JavaScript murni dengan 4 menu utama:
  - `Beranda` (`index.html`): Ikhtisar ancaman AI dan alur kerja sistem.
  - `Deteksi Citra` (`deteksi.html`): Form unggah drag-and-drop, validasi format/ukuran, dan pratinjau hasil.
  - `Cara Kerja` (`panduan.html`): Panduan teknis dan alur pemrosesan tensor.
  - `Tentang Kami` (`tentang.html`): Profil Kelompok 8 dan spesifikasi arsitektur.
- **Validasi Berkas Ketat**: Mendukung format `.jpg`, `.jpeg`, `.png` dengan batas maksimal 10 MB.
- **Pemrosesan In-Memory**: Gambar diproses secara sementara di RAM/VRAM tanpa disimpan permanen di server untuk privasi pengguna.

---

## Struktur Direktori

```text
DeepGuard-AI/
├── models/
│   └── resnet18_model.pt       # Bobot model ResNet-18 terlatih (State Dict PyTorch)
├── static/
│   ├── css/
│   │   └── style.css           # Lembar gaya CSS utama
│   └── img/                    # Aset grafis antarmuka & logo
├── templates/
│   ├── index.html              # Halaman Beranda
│   ├── deteksi.html            # Halaman Deteksi Citra
│   ├── panduan.html            # Halaman Cara Kerja / Panduan
│   └── tentang.html            # Halaman Profil & Info Sistem
├── app.py                      # Server backend Flask & REST API (/api/predict)
├── requirements.txt            # Dependensi pustaka Python
├── test_api.py                 # Skrip pengujian end-to-end API inferensi
├── train_cuda.py               # Skrip pelatihan model pada akselerasi GPU CUDA
└── README.md                   # Dokumentasi proyek
```

---

## Persyaratan Sistem

- **Python**: 3.10, 3.11, atau 3.12
- **Hardware**: CPU atau GPU NVIDIA (disarankan GPU dengan CUDA untuk inferensi super cepat <50ms)
- **Library**: Flask, PyTorch, Torchvision, Pillow

---

## Panduan Instalasi & Menjalankan Aplikasi

### 1. Clone Repositori

```bash
git clone https://github.com/mushlihularif-gif/NAMA_REPO_ANDA.git
cd NAMA_REPO_ANDA
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi Disarankan)

```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux / macOS
python3 -m venv env
source env/bin/activate
```

### 3. Pasang Dependensi

```bash
pip install -r requirements.txt
```

> **Catatan Pengguna GPU NVIDIA**: Untuk mengaktifkan akselerasi CUDA pada PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 4. Jalankan Server Flask

```bash
python app.py
```

Buka peramban (browser) dan akses:
```
http://127.0.0.1:5000/deteksi.html
```

---

## Tim Pengembang (Kelompok 8)

Mata Kuliah: **Artificial Intelligence for Security**  
Program Studi: **Teknik Informatika - Semester 5**
