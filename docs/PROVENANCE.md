# Pemeriksaan Content Credentials (C2PA)

## Masalah dan perbaikan

Klasifikasi visual Community Forensics masih gagal pada infografik AI pengguna:
skor AI sekitar 0,00918% dan kelas visual REAL. Berkas asli ternyata menyertakan
manifest C2PA yang ditandatangani OpenAI Media Service, dengan sumber
`trainedAlgorithmicMedia`. Aplikasi sebelumnya membuang bukti ini ketika hanya
meneruskan gambar RGB ke model. Ilustrasi mesin pesawat sebelumnya juga memiliki
deklarasi AI valid yang sama, meskipun skor visualnya sekitar 39,86%.

Sekarang ada dua hasil terpisah. `model_label` dan `prob_fake` tetap berasal dari
model piksel tanpa modifikasi. `label` dapat menjadi `AI_DISCLOSED` dengan
`decision_basis=verified_c2pa` jika bukti asal lolos pemeriksaan. UI menampilkan
"Penggunaan AI tercatat", penandatangan, dan penjelasan bahwa skor visual bukan
dasar keputusan tersebut. Kredensial membuktikan adanya deklarasi yang ditandatangani
dan terkait dengan berkas, bukan menjamin kebenaran setiap pernyataan penciptanya.

## Syarat penerimaan

- SDK resmi `c2pa-python==0.39.0` membaca byte unggahan asli di memori.
- Hanya manifest aktif; tidak mencari kata AI di metadata, nama file, training
  permissions, thumbnail, atau manifest bahan lain yang tidak terkait.
- `validation_state=Trusted`; tidak ada kegagalan aktif; harus ada keberhasilan
  pemeriksaan tanda tangan, trust penandatangan, dan hard binding hash gambar.
- Deklarasi jenis sumber AI harus berada di assertion `c2pa.actions` atau
  `c2pa.actions.v2`, dengan URI IPTC yang didukung secara eksplisit.
- Kepercayaan hanya dari snapshot daftar resmi C2PA di `trust/c2pa-official.pem`.
  URL commit, hash, dan tanggal pengambilan tersedia di `trust/source.json`.
  Tidak menambahkan sertifikat yang diambil dari gambar sebagai trusted root.
- Hash bundle diperiksa saat membaca. Bundle hilang/berubah tidak mematikan
  pemeriksaan tanda tangan; status menjadi unavailable, tanpa keputusan C2PA.

## Batas dan privasi

Tidak ada remote-manifest fetch, OCSP fetch, atau CAWG identity lookup. Network
allow-list SDK kosong. Unggahan tidak dikirim ke layanan eksternal. SDK memakai
daftar trust lokal sehingga pencabutan sertifikat terbaru tidak diperiksa online.
Informasi timestamp dapat tidak tepercaya; aplikasi tidak mengesahkan tanggal
pembuatan. Pembaruan daftar trust harus ditinjau terpisah, mengunduh ulang dari
sumber resmi, mencatat revisi/hash baru, dan menjalankan uji regresi.

Kredensial hilang, tidak dipercaya, atau rusak tidak menjadi bukti AI maupun
non-AI. Deklarasi digitalCapture tidak memaksa hasil REAL. Menghapus metadata,
screenshot, atau re-encode dapat menghilangkan C2PA; saat itu model visual tetap
memiliki keterbatasan sebelumnya. Ini tidak melatih model dan tidak menaikkan
angka benchmark visual 84 gambar. Edit lokal hanya dapat tercatat melalui
kredensial yang sesuai; kemampuan deteksi visual edit lokal belum tervalidasi.

## Pengujian

`tests/test_provenance.py` membuat CA dan kunci sementara khusus tes di memori,
bukan menaruh kredensial pengguna atau private key di repositori. Tes menandatangani
PNG sintetis dan memeriksa:

- deklarasi AI penuh dan komposit diterima dengan CA tes yang secara eksplisit
  dipercaya hanya di lingkup tes;
- CA tersebut ditolak oleh konfigurasi produksi;
- menyalin kredensial ke piksel lain ditolak oleh hash binding;
- kredensial yang dihilangkan tidak menghasilkan bukti AI;
- deklarasi kamera tidak menjadi jaminan non-AI;
- kredensial bahan lain tidak memaksa gambar aktif menjadi AI;
- kerusakan bundle trust tidak menyebabkan bypass verifikasi.

Tes API memeriksa keputusan C2PA tanpa memalsukan skor visual; tes frontend
memeriksa judul, skor terpisah, dan rendering nama penandatangan sebagai teks.
Dua kasus pengguna diuji lokal saja; gambar/manifest lengkapnya tidak dipublikasikan.

Sumber:
[SDK dan penggunaan stream](https://github.com/contentauth/c2pa-python/blob/main/docs/usage.md),
[daftar resmi C2PA](https://github.com/c2pa-org/conformance-public/tree/main/trust-list),
[penjelasan trust](https://opensource.contentauthenticity.org/docs/conformance/trust-lists/).
