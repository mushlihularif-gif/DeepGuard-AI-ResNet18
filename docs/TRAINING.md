# Training, pengujian, dan cloud

## Memakai aplikasi tidak memerlukan dataset

Flask hanya membutuhkan kode, library, dan checkpoint sekitar 45 MB. Dataset
dipakai saat training/evaluasi, bukan diunduh ketika pengguna mengunggah gambar.
Versi ini tidak melatih ulang saat startup. File `calibration_anchors.py` dan
`models/calibrated_*.flag` dari versi lama tidak lagi dipakai.

## Reproduksi eksperimen lokal

Instal `pip install -r requirements-train.txt`. Dataset yang dipakai saat perbaikan
sudah tersedia di laptop; tidak ada unduhan dataset tambahan. Jalankan dari root:

```powershell
python prepare_data.py --tiny-train E:\train_clean_saved --tiny-val E:\val_resmi_saved --modern-cache "$env:USERPROFILE\.cache\huggingface\datasets\Hemg___ai-generated-vs-real-images-datasets"
python train.py --tiny-train E:\train_clean_saved --tiny-val E:\val_resmi_saved --modern-cache "$env:USERPROFILE\.cache\huggingface\datasets\Hemg___ai-generated-vs-real-images-datasets" --epochs 6
```

`prepare_data.py` membuat `work/manifest.json`. Bobot kandidat masuk ke
`models/candidate.pt`, bukan menimpa model aplikasi. Untuk menggunakan manifest
eksperimen yang disertakan, tambahkan `--manifest reports/data_manifest.json` pada
perintah train; fingerprint dataset harus cocok. Dataset publik yang berubah
perlu manifest baru. Hasil tidak dijamin identik bit demi bit pada GPU berbeda.

Pembagian: 12.000 train, 800 validation (pemilihan epoch), 800 calibration
(temperature scaling), 1.600 test. Setiap sumber dan kelas seimbang. Hash gambar
RGB 64×64 menolak duplikat normalisasi lintas split; ini bukan deteksi menyeluruh
untuk near-duplicate atau gambar dari adegan/orang yang sama. Nilai test tidak
dipakai untuk memilih epoch. Setelah melihat test, jangan terus menyesuaikan model
dan tetap menyebutnya test independen: kumpulkan holdout baru.

Label Tiny-GenImage: real=0, fake=1. Label Hemg: AiArtData=0 → FAKE=1,
RealArt=1 → REAL=0. Nama kelas asing menyebabkan error, bukan ditebak sebagai AI.
Gambar korup menyebabkan error saat training, tidak diganti tensor nol berlabel REAL.

## Training di cloud tanpa dataset besar di laptop

Colab menjalankan Python di VM cloud, bukan laptop. Unduh kembali checkpoint dan
laporan saja setelah selesai. GPU, batas waktu, dan storage VM tidak dijamin tetap;
simpan hasil sebelum runtime berakhir. Lihat [FAQ resmi Colab](https://research.google.com/colaboratory/faq.html).

Contoh alur Colab (belum dijalankan di akun cloud pengguna):

```python
!git clone https://github.com/mushlihularif-gif/DeepGuard-AI-ResNet18.git
%cd DeepGuard-AI-ResNet18
!pip install -r requirements-train.txt
```

Untuk menghemat storage VM juga, ambil subset seimbang lewat streaming. Data tetap
melewati jaringan; streaming tidak berarti tanpa bandwidth/RAM/cache. Sampel
terpilih disimpan di VM agar training beberapa epoch tidak mengunduh ulang.
Ini contoh eksperimen baru, bukan reproduksi persis benchmark di repositori:

```python
from datasets import load_dataset, Dataset, Features, Image as HFImage, ClassLabel
from modeling import label_mapping, rgb_image
import io

def collect(repo, split, per_class, output):
    stream = load_dataset(repo, split=split, streaming=True)
    mapping = label_mapping(stream.features['label'].names)
    rows, counts, hashes = [], [0, 0], set()
    import hashlib
    # Decode hanya gambar yang akan diambil; buffer shuffle bukan shuffle global.
    stream = stream.cast_column('image', HFImage(decode=False)).shuffle(seed=20261117, buffer_size=512)
    for item in stream:
        label = mapping[int(item['label'])]
        if counts[label] >= per_class:
            continue
        from PIL import Image
        encoded = item['image']
        with Image.open(io.BytesIO(encoded['bytes']) if encoded.get('bytes') is not None else encoded['path']) as raw:
            image = rgb_image(raw)
            digest = hashlib.sha256(image.resize((64, 64)).tobytes()).hexdigest()
            if digest in hashes:
                continue
            hashes.add(digest)
            # Simpan subset resolusi terbatas di VM, bukan seluruh sumber.
            image.thumbnail((768, 768))
            buffer = io.BytesIO()
            image.save(buffer, 'PNG')
        rows.append({'image': {'bytes': buffer.getvalue(), 'path': None}, 'label': label})
        counts[label] += 1
        if min(counts) >= per_class:
            break
    if min(counts) < per_class:
        raise ValueError(f'Sumber tidak mencukupi: {counts}')
    features = Features({'image': HFImage(), 'label': ClassLabel(names=['real', 'fake'])})
    Dataset.from_list(rows, features=features).save_to_disk(output)

collect('TheKernel01/Tiny-GenImage', 'train', 3300, '/content/data/tiny_train')
collect('TheKernel01/Tiny-GenImage', 'validation', 1000, '/content/data/tiny_val')
collect('Hemg/AI-Generated-vs-Real-Images-Datasets', 'train', 4200, '/content/data/hemg')
```

```python
!python prepare_data.py --tiny-train /content/data/tiny_train --tiny-val /content/data/tiny_val --modern-cache /content/data/hemg --legacy-modern-prefix 0
!python train.py --tiny-train /content/data/tiny_train --tiny-val /content/data/tiny_val --modern-cache /content/data/hemg --epochs 6
from google.colab import files
files.download('models/candidate.pt')
files.download('reports/evaluation.json')
```

Bandingkan metrik per sumber/kelas sebelum mengganti `models/resnet18_model.pt`,
lalu restart Flask. Jangan mengunggah dataset/gambar pribadi ke GitHub. Panduan
streaming: [dokumentasi Hugging Face](https://huggingface.co/docs/datasets/stream).

## Yang dibutuhkan untuk edit sebagian gambar

Ganti baju/wajah/latar dengan AI masuk kelas positif meskipun bagian lain masih
foto kamera. Dataset saat ini belum memvalidasi kemampuan tersebut. Perlu pasangan
foto asli–hasil edit yang asalnya diketahui, ragam alat/edit/kompresi, dan split
berdasarkan gambar sumber (semua turunan masuk split yang sama). Pertimbangkan model
lokalisasi yang dilatih dengan mask, lalu ukur false positive pada foto kamera.
Jangan menganggap crop tertinggi, EXIF, nama file, atau sudut terang sebagai bukti AI.

Untuk uji eksternal, siapkan CSV `path,label,group`, dengan group misalnya `camera`,
`generated`, `ai_edited`, lalu jalankan:

```bash
python evaluate_uploads.py data/ground_truth.csv --output work/external_evaluation.json
```

Label berasal dari proses pembuatan gambar yang diketahui, bukan dugaan dari tampilannya.
Laporan memisahkan keputusan salah, benar, ragu, dan error. Gambar tidak dipakai untuk
training otomatis. Jangan mengklaim akurasi universal berdasarkan subset ini.
