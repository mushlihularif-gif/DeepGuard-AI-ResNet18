# Pilot lintas sumber dan pergantian mesin

Catatan lanjutan: angka di laporan ini mengukur model visual saja. Aplikasi kini
juga memeriksa [kredensial C2PA](../docs/PROVENANCE.md). Ilustrasi pesawat dan
infografik pengguna memiliki deklarasi AI valid; keputusan bukti asal kini
`AI_DISCLOSED`, sementara skor model visual dan metrik di bawah tetap sama.

Keluhan pengguna terbukti pada data pembanding: hasil ResNet18 pada subset
latihannya sendiri terlalu optimistis untuk penggunaan di luar sumber tersebut.
Pemetaan label sudah benar, tetapi memperbaiki bug label tidak otomatis membuat
model mampu mengenali semua generator baru. Menambah jumlah data dari sumber
sempit juga bukan jaminan generalisasi.

## Protokol

- Sumber publik: `OwensLab/CommunityForensics-Eval`, config `default`, split
  `CompEval`, label penulis 0=non-AI dan 1=AI.
- Seed 20261118; 32 offset tersebar pada 51.836 baris, empat baris tiap permintaan.
  Offset ditetapkan sebelum kedua model diuji, bukan berdasarkan skor.
- 21 permintaan berhasil: 84 gambar, 32 non-AI dan 52 AI. Sebelas permintaan
  gagal karena timeout/error server; daftar offset/error ada di JSON protokol.
  Filter NSFW, duplikat RGB 64x64 dan overlap dengan semua split lokal tersedia;
  tidak ada dari 84 gambar yang berhasil diterima dikeluarkan oleh filter.
- Nama kelompok `model_name` dapat mencakup pasangan gambar non-AI. Metrik per
  kelompok bukan otomatis metrik khusus generator tanpa melihat labelnya.
- Bobot, preprocessing dan ambang tetap. Tidak ada latihan pada pilot atau
  gambar pribadi pengguna. Tidak ada aturan nama file, EXIF, atau lookup hash
  untuk mengubah prediksi. Hash gambar dipakai hanya untuk integritas evaluasi.
- Pilot dipakai untuk memilih mesin, sehingga bukan set konfirmasi akhir.
  Seluruh korpus pralatih Community Forensics tidak tersedia lokal untuk audit
  overlap menyeluruh. Tidak mengklaim semua 84 gambar independen dari pralatihnya.

## Hasil

| Input / mesin | Benar pada ambang 0,5 | Non-AI → AI | AI → non-AI |
| --- | ---: | ---: | ---: |
| Sumber / ResNet18 | 48/84 | 11/32 | 25/52 |
| Sumber / Community Forensics | 77/84 | 0/32 | 7/52 |
| JPEG 70, maks. 512 / ResNet18 | 48/84 | 11/32 | 25/52 |
| JPEG 70, maks. 512 / Community Forensics | 67/84 | 0/32 | 17/52 |

UI memakai area ragu 0,2–0,8 yang sama untuk kedua model, bukan ambang 0,5.
Pada input sumber, Community Forensics: 74 benar, 4 salah, 6 ragu. Pada versi
kompresi: 61 benar, 12 salah, 11 ragu. Tidak ada tuning untuk menghilangkan kesalahan.
Angka 0/32 false positive tidak membuktikan false positive nol secara umum.

Kompresi menurunkan kinerja. Ini relevan untuk foto yang diteruskan lewat chat,
meskipun transformasi uji tidak meniru semua aplikasi chat. Pengujian ini tidak
mencakup semua kamera, generator, screenshot, ilustrasi, atau perubahan lokal.

Ilustrasi mesin pesawat pengguna, diuji terpisah dari pilot, menghasilkan skor
AI sekitar 0,3986 pada model baru dan tetap UNCERTAIN. Tidak ada berkas pribadi
yang disertakan di Git atau digunakan untuk training. Mengganti baju dengan AI
memerlukan evaluasi edit lokal; model klasifikasi gambar utuh ini belum divalidasi
untuk tugas tersebut. Center crop juga dapat membuang bagian yang diedit di tepi.

## Implementasi dan reproduksi

Mesin utama: Community Forensics ViT-S/16, input384, checkpoint resmi
`OwensLab/commfor-model-384`, revisi
`6076002bf0d9dd37537f965ee2f06f826c333b61`, SHA-256
`b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`.
Resize sisi pendek440, center crop384, normalisasi ImageNet sesuai kode penulis.
Skor sigmoid mentah diadaptasi dengan softmax([0,logit]) yang identik secara
matematis; bukan pembalikan kelas atau kalibrasi baru.

ResNet18 tetap dipertahankan. Laporan JSON menyimpan kedua hash model, semua
label/skor/ID baris/hash gambar; tidak menyimpan piksel atau nama berkas pribadi.
`fetch_cross_domain.py` mengambil hanya sampel yang tercatat dan memverifikasi
hash; `evaluate_cross_domain.py` mengulang evaluasi dengan bobot tetap.
Gunakan README untuk perintah lengkap. Perubahan dataset publik dapat membuat
reproduksi gagal dengan pesan hash mismatch, bukan diam-diam mengganti sampel.

Sumber: [kode dan paper resmi](https://github.com/JeongsooP/Community-Forensics),
[model](https://huggingface.co/OwensLab/commfor-model-384),
[dataset evaluasi](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval).

Pekerjaan lanjutan untuk klaim lebih kuat: uji terkunci yang lebih besar dengan
kamera beragam dan generator yang benar-benar ditahan dari pemilihan model;
benchmark pasangan sebelum/sesudah edit lokal; kalibrasi pada split terpisah.
Mengubah label atau ambang berdasarkan satu keluhan bukan solusi generalisasi.
