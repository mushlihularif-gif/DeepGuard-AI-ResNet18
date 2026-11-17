# Audit perbaikan DeepGuard — 17 November 2026

## Masalah yang benar-benar ditemukan

Kode lokal awal: `24f4960`; perubahan GitHub `f6ce766` diintegrasikan sebelum
perbaikan. Bobot kedua versi disimpan lokal sebagai baseline untuk perbandingan.
Gambar kegagalan pengguna sudah dihapus, sehingga kegagalan spesifik tersebut
tidak dapat direproduksi. Perbaikan tidak diklaim menjamin semua gambar baru benar.

1. `ModernDatasetWrapper` lama menganggap nama kelas apa pun di luar daftar kecil
   sebagai AI. `RealArt` tidak ada pada daftar itu, sehingga akan dipetakan salah.
   Dalam run V3 yang tersedia, pemilihan 20.000 baris awal Hemg justru seluruhnya
   `AiArtData`: foto/non-AI dari sumber tambahan tidak pernah ikut dipilih. Jadi
   bug alias RealArt dan bias pengambilan baris merupakan dua masalah berbeda;
   ini bukan bukti bahwa seluruh output kelas model perlu dibalik.
2. Skrip hybrid lokal melabeli file berdasarkan nama, mengulang gambar berkali-kali,
   kemudian menguji beberapa file yang sama. Hasil itu bukan pengujian generalisasi.
3. Versi GitHub menggabungkan skor model dengan aturan nama file, EXIF, resolusi,
   residual blur, dan sudut terang. Tidak ada evaluasi terpisah yang membuktikan
   bobot aturan tersebut. EXIF kamera juga dapat tetap ada setelah edit AI.
4. Startup bisa menjalankan pelatihan anchor jika flag hilang. Checkpoint dapat
   berubah hanya karena aplikasi dibuka, tanpa evaluasi dan identitas model jelas.
5. Antarmuka menyebut foto terverifikasi, FFT, dan noise sensor terukur padahal
   inferensi tidak menghasilkan bukti pemeriksaan itu. Demo memakai angka tetap.
6. Dataset klasifikasi gambar utuh belum memvalidasi manipulasi lokal seperti
   ganti pakaian. Resize ke 224×224 dapat mengurangi detail edit kecil; mengganti
   mean dengan maksimum crop saja tidak membuktikan peningkatan kemampuan.

## Perbaikan

Model baru dimulai dari ImageNet, bukan bobot hybrid/anchor. Pemetaan label dan
preprocessing dipakai bersama trainer dan server. Kelas/sumber diseimbangkan,
duplikat RGB ternormalisasi ditolak, serta data train, validation, calibration,
test dipisah. Manifest publik berisi indeks dan hash, tidak menyalin gambar pribadi.

Enam epoch ditetapkan sebelum melihat test. Epoch dipilih dengan balanced accuracy
validation. Satu temperatur dipilih dengan cross-entropy calibration. Test baru
dibuka setelah pemilihan itu selesai. Ambang 0,2/0,8 ditetapkan sebagai area ragu;
tidak dipromosikan sebagai batas error terjamin. Model menolak bobot tanpa metadata
dan tidak pernah melatih/menebak otomatis ketika startup gagal.

Skor sekarang murni hasil ResNet18 terkalibrasi, tanpa bonus nama file/EXIF/sudut
gambar. REAL ditampilkan sebagai kecenderungan non-AI. UNCERTAIN menahan keputusan.
Klaim diagnostik palsu dan hasil simulasi tetap sudah dihapus. Metrik UI bersumber
dari checkpoint aktif. API mengembalikan hash model untuk penelusuran.

## Hasil terukur

| Bobot / kondisi | Seluruh 1.600 gambar | Tiny (800) | Hemg (800) |
|---|---:|---:|---:|
| Lama lokal (24f4960) | 64,25% | 78,50% | 50,00% |
| Lama GitHub (f6ce766) | 62,38% | 73,38% | 51,38% |
| Baru, epoch 3 terpilih | **86,19%** | **82,88%** | **89,50%** |

Temperatur kalibrasi: 1,139872. Dengan area ragu, 1.156/1.600 (72,25%)
sampel mendapat keputusan; 1.089/1.156 (94,20%) keputusan tersebut benar,
67 salah, dan 444 sampel abstain. Ini tidak berarti akurasi universal 94,20%.
Resize maksimum 512 + JPEG kualitas 70 menghasilkan akurasi 85,00% pada gambar
uji yang sama; cakupan keputusan 71,63%, akurasi keputusan 93,46%.

Enam belas regression test Python dan tes frontend lolos. Unggahan HTTP nyata
memakai checkpoint dengan hash yang sesuai, dan browser menampilkan respons yang
sama. Sampel AI publik pertama dari manifest menghasilkan UNCERTAIN (skor AI
38,08%), menunjukkan model masih dapat kesulitan; sampel tersebut tidak diganti
untuk membuat demonstrasi tampak sempurna.

## Cara membaca hasil

Metrik rinci tersedia di `evaluation.json`; indeks/hash di `data_manifest.json`.
Baris baseline membandingkan **ResNet18 global single-view** dari checkpoint lama,
bukan endpoint lama yang mencampur heuristik filename/EXIF dan flip. Ini mengisolasi
kualitas bobot pada masukan yang identik. Tidak ada klaim perbandingan menyeluruh
terhadap semua aturan endpoint lama.

Accuracy biasa memakai argmax untuk seluruh sampel. API memakai area ragu:
`decision_coverage` adalah bagian sampel yang mendapat REAL/FAKE,
`decided_accuracy` adalah akurasi hanya di bagian itu. Jangan melaporkan
decided_accuracy tanpa coverage atau menganggap abstain sebagai prediksi benar.

`robustness.json` menggunakan **gambar test yang sama**, diperkecil maksimal sisi
512 lalu JPEG kualitas 70. Itu bukan 1.600 gambar independen tambahan dan bukan
replikasi seluruh pemrosesan WhatsApp.

## Batas kesimpulan

- Tiny validation pernah dipakai training pipeline model lama; baseline Tiny tidak
  independen. Model baru tidak dilatih dengan split tersebut.
- Hemg `RealArt` mencakup seni non-AI. Tidak sama dengan korpus foto kamera terverifikasi.
- Hash RGB 64×64 tidak menghilangkan seluruh near-duplicate atau kebocoran adegan.
- Generator baru dan foto dengan edit AI lokal belum memiliki holdout terverifikasi.
- Kalibrasi di dua sumber ini tidak menjamin kalibrasi pada gambar internet lain.
- Cloud mengubah lokasi komputasi dan penyimpanan, bukan otomatis kualitas model.

Uji berikutnya harus memakai foto kamera, generasi AI, serta pasangan asli–edit
yang asalnya diketahui; kelompokkan seluruh turunan foto yang sama dalam satu split.
`evaluate_uploads.py` sudah menyediakan laporan per kelompok untuk pengujian itu.

## Sumber

- [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)
- [Hemg AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- [Community Forensics, CVPR 2025](https://arxiv.org/abs/2411.04125): penelitian keragaman generator; modelnya tidak dipasang di proyek ini.
- [SIDA / SID-Set](https://github.com/hzlsaber/SIDA): contoh benchmark yang membedakan gambar sintetis dan manipulasi lokal; belum digunakan pada eksperimen ini.
