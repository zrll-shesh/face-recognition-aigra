# Face Recognition Attendance System

Proof of Concept (PoC) sistem absensi karyawan berbasis face recognition, dikembangkan
sebagai bagian dari tugas magang AI Engineer. Dokumen ini merangkum alur kerja, model
yang digunakan, hasil analisis dari notebook, serta rekomendasi untuk tahap selanjutnya.

---

## 1. Ringkasan Kebutuhan

PIC meminta sistem face recognition untuk absensi karyawan dengan alur berikut:

```
Karyawan -> Buka aplikasi kantor -> Scan wajah -> Sistem hitung confidence score
   -> Tampilkan nama/identitas yang teridentifikasi
   -> "Apakah ini benar [Nama]?"
        -> Ya, benar                -> Absensi tercatat (auto-verified)
        -> Bukan / confidence < 82% -> Verifikasi manual
```

Requirement utama dari PIC:

- Model akurat untuk mengenali wajah karyawan.
- Ada confidence score, dengan ambang batas (threshold) untuk auto-verifikasi.
- Hasil akhirnya bisa langsung dipasang ke backend aplikasi kantor.

Catatan revisi: threshold awal yang diminta PIC adalah 90 persen. Setelah melihat hasil
kalibrasi pada data karyawan (dijelaskan di bagian 4), PIC merevisi threshold menjadi
82 persen. Seluruh angka di dokumen ini sudah menggunakan threshold 82 persen, kecuali
disebutkan sebagai perbandingan historis.

---

## 2. Alur Kerja dan Inisiatif yang Dilakukan

Berikut proses yang dijalani dari awal sampai deliverable ini, termasuk bagian yang
dikerjakan di luar permintaan awal PIC sebagai bentuk inisiatif.

### 2.1 Riset pendekatan (sebelum coding)

Requirement awal PIC hanya "cari atau bikin model untuk face recognition". Sebelum
langsung eksekusi, dilakukan analisis pendekatan mana yang paling tepat.

| Opsi | Kelayakan |
|---|---|
| Training model dari nol | Tidak layak. Butuh dataset jutaan wajah dan resource komputasi besar, tidak realistis untuk kebutuhan internal kantor. |
| Fine-tuning model pretrained | Tidak diperlukan. Data enrollment karyawan (beberapa foto per orang) terlalu sedikit untuk fine-tuning, berisiko overfitting. |
| Pakai model pretrained apa adanya, fokus di pipeline dan kalibrasi | Dipilih. Model face recognition modern sudah dilatih di jutaan wajah dan terbukti general ke wajah baru yang belum pernah dilihat. Pekerjaan inti ada di pipeline (deteksi, alignment, embedding, matching) dan kalibrasi threshold sesuai data nyata. |

Kesimpulan: tidak perlu training atau fine-tuning. Cukup gunakan model pretrained
sebagai mesin pengenal wajah, lalu bangun sistem enrollment, pencocokan, dan kalibrasi
di atasnya.

### 2.2 Implementasi pipeline

Dibangun pipeline lengkap dengan tiga tahap standar industri untuk face recognition:

1. Face Detection: mencari lokasi wajah dan titik landmark (mata, hidung, mulut) di dalam gambar.
2. Face Alignment: merapikan posisi wajah berdasarkan landmark sebelum diproses, supaya hasil embedding konsisten walau sudut wajah sedikit berbeda.
3. Face Recognition (Embedding): mengubah wajah menjadi vektor angka 512 dimensi yang merepresentasikan identitas wajah tersebut.

Proses pencocokan (matching) dilakukan dengan menghitung cosine similarity antara
embedding wajah yang baru discan dengan embedding yang tersimpan di database, lalu
similarity itu dipetakan menjadi confidence score dalam persen.

### 2.3 Data enrollment

Dikumpulkan foto dari 3 karyawan sebagai data uji awal (ghani, nadia, nazril), masing-masing
5 foto per orang dengan variasi angle/kondisi. Ini digunakan untuk membangun database
embedding dan menguji apakah pipeline benar-benar bisa membedakan satu orang dengan
orang lain.

### 2.4 Inisiatif tambahan: analisis dan kalibrasi mendalam

Di luar permintaan awal PIC yang hanya meminta pencarian model, dilakukan analisis
tambahan lewat notebook (`notebooks/face_recognition_pipeline.ipynb`) untuk memastikan
sistem ini bisa dipertanggungjawabkan secara data, bukan asal pakai model:

- Analisis distribusi similarity antara wajah yang sama (genuine) vs wajah berbeda (impostor).
- Kalibrasi threshold berbasis data nyata, menggunakan metode standar industri biometrik: False Accept Rate, False Reject Rate, dan Equal Error Rate.
- Pemetaan cosine similarity menjadi confidence score persen, dikalibrasi ke requirement dari PIC.
- Visualisasi ruang embedding (PCA, t-SNE) untuk melihat seberapa rapi sistem mengelompokkan tiap identitas.
- Simulasi pengujian end-to-end (leave-one-out testing) dan confusion matrix, untuk memperkirakan performa nyata sebelum dipasang ke backend.
- Dashboard Streamlit dan contoh backend API (FastAPI) sebagai referensi implementasi, sehingga hasil kerja ini langsung bisa dites secara visual dan siap diintegrasikan.

Tujuan bagian ini bukan formalitas, tapi untuk menjawab pertanyaan penting sebelum
sistem dipasang ke backend produksi: seberapa yakin kita bahwa sistem ini benar-benar
bisa membedakan karyawan satu dengan yang lain, dan di titik confidence berapa sistem
mulai tidak bisa dipercaya. Analisis inilah yang kemudian menjadi dasar keputusan PIC
untuk merevisi threshold dari 90 persen menjadi 82 persen (lihat bagian 4.3 dan 5).

---

## 3. Model yang Digunakan

### 3.1 Sumber

Model diambil dari repository resmi InsightFace, tim riset yang mempublikasikan metode
ArcFace, bukan model pihak ketiga atau hasil training sendiri.

- Repository: https://github.com/deepinsight/insightface
- File yang diunduh otomatis oleh library: https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip

### 3.2 Paket model: buffalo_l

buffalo_l bukan satu model tunggal, melainkan bundle berisi 5 model kecil yang saling
melengkapi.

| File | Fungsi | Arsitektur | Dipakai di pipeline |
|---|---|---|---|
| det_10g.onnx | Deteksi lokasi wajah dan landmark | SCRFD (detector) | Ya |
| w600k_r50.onnx | Generate embedding 512 dimensi, inti recognition | ResNet-50 + ArcFace loss | Ya, ini yang utama |
| genderage.onnx | Prediksi gender/umur | - | Tidak |
| 2d106det.onnx | 106 titik landmark wajah detail | - | Tidak |
| 1k3d68.onnx | Landmark 3D | - | Tidak |

### 3.3 Detail model utama: w600k_r50

Ini bagian yang paling penting dijelaskan ke PIC karena inilah otak pengenal wajahnya.

- Arsitektur backbone: ResNet-50.
- Metode training: ArcFace (Additive Angular Margin Loss), metode dari paper "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (Deng et al., 2019), ditulis oleh tim yang sama yang mempublikasikan InsightFace. Metode ini termasuk yang paling banyak dipakai di industri untuk face recognition karena menghasilkan embedding dengan pemisahan antar identitas yang tajam.
- Dataset training: WebFace600K, dataset publik skala besar berisi ratusan ribu identitas wajah.
- Output: vektor embedding 512 dimensi per wajah. Dua wajah dibandingkan dengan menghitung cosine similarity antar vektor embeddingnya.

### 3.4 Akurasi berdasarkan benchmark publik

Angka berikut adalah hasil benchmark resmi dari InsightFace di dataset akademik standar,
bukan hasil pengujian internal kita.

- LFW: sekitar 99.85 persen.
- IJB-C (TAR pada FAR = 1e-4): sekitar 96 sampai 97.5 persen.

Catatan penting untuk PIC: angka ini adalah hasil di dataset benchmark publik, bukan di
populasi karyawan kantor. Itu sebabnya kalibrasi ulang menggunakan foto karyawan asli
(dijelaskan di bagian 4) tetap wajib dilakukan sebelum menentukan threshold final.

### 3.5 Kenapa memilih buffalo_l, bukan varian lain

InsightFace menyediakan beberapa varian model dengan trade off berbeda antara akurasi
dan kecepatan.

- buffalo_sc dan buffalo_s: ringan, untuk perangkat mobile/edge dengan resource terbatas.
- buffalo_m: ukuran menengah.
- buffalo_l: akurasi lebih tinggi, ditujukan untuk server. Dipilih untuk proyek ini.
- antelopev2: paling besar dan paling akurat, tapi lebih berat secara komputasi.

Karena sistem ini akan berjalan di backend server, bukan langsung di perangkat HP
karyawan, buffalo_l dipilih sebagai titik seimbang antara akurasi tinggi dan kecepatan
proses yang masih wajar untuk skala jumlah karyawan kantor.

---

## 4. Hasil dan Insight dari Notebook

Notebook face_recognition_pipeline.ipynb dijalankan penuh menggunakan data 3 karyawan
(ghani, nadia, nazril), masing-masing 5 foto. Bagian ini sudah menggunakan hasil run
kedua, setelah threshold direvisi PIC menjadi 82 persen.

### 4.1 Distribusi similarity: wajah sama vs wajah berbeda

| Kategori | Jumlah pasangan | Rata-rata similarity | Std deviasi |
|---|---|---|---|
| Genuine, wajah orang yang sama | 30 pasangan | 0.741 | 0.065 |
| Impostor, wajah orang berbeda | 75 pasangan | 0.076 | 0.046 |

Insight: ada jarak yang sangat lebar antara similarity wajah yang sama, sekitar 0.74,
dan wajah berbeda, sekitar 0.08. Ini menunjukkan model mampu membedakan ketiga karyawan
ini dengan sangat jelas, tidak ada tumpang tindih yang berarti antara kedua distribusi.
Nilai ini tidak berubah dari run sebelumnya karena murni berasal dari data wajah dan
model, tidak dipengaruhi oleh perubahan threshold confidence.

### 4.2 Kalibrasi threshold: FAR, FRR, Equal Error Rate

- Equal Error Rate (EER): 0.000 persen, tercapai di similarity threshold 0.192.
- Threshold similarity yang direkomendasikan sistem: 0.242, yaitu EER ditambah margin keamanan.

Insight: EER 0 persen berarti, pada data uji ini, ada titik ambang di mana sistem tidak
pernah salah menerima orang yang salah (false accept) maupun salah menolak orang yang
benar (false reject). Nilai EER dan threshold similarity ini adalah properti dari
sebaran data wajah itu sendiri, sehingga tidak berubah walau target confidence persen
diubah dari 90 menjadi 82 persen. Yang berubah hanya cara similarity dipetakan menjadi
angka persen di bagian 4.3.

### 4.3 Pemetaan similarity ke confidence score, revisi ke 82 persen

- Batas bawah kalibrasi (similarity_low_bound): 0.004.
- Batas atas kalibrasi (similarity_high_bound): 0.853.
- Similarity yang dibutuhkan untuk mencapai confidence 82 persen: 0.700.
- Sebagai perbandingan, similarity yang dibutuhkan untuk confidence 90 persen (angka lama): 0.768.

Insight: rata-rata similarity genuine adalah 0.741. Di threshold lama (90 persen),
similarity 0.741 ini berada tepat di bawah syarat 0.768, sehingga banyak percobaan
genuine gagal auto-verifikasi. Di threshold baru (82 persen), syarat similarity turun
menjadi 0.700, yang sekarang berada di bawah rata-rata similarity genuine (0.741).
Artinya sebagian besar percobaan absen dari karyawan yang benar sekarang secara
konsisten berada di atas syarat minimum, bukan lagi tepat di garis batas.

### 4.4 Simulasi pengujian end to end, leave-one-out testing

Setiap foto diuji sebagai wajah yang discan, dicocokkan ke sisa foto lain di database,
menggunakan metode leave-one-out, yaitu metode standar untuk data terbatas.

| Metrik | Threshold 90% (lama) | Threshold 82% (revisi) |
|---|---|---|
| Akurasi auto-verifikasi | 60.00 persen | 93.33 persen |
| Tingkat verifikasi manual | 40.00 persen | 6.67 persen |
| Tingkat salah kenal orang (false identification) | 0 persen | 0 persen |

Insight paling penting dari simulasi ini: pada kedua threshold, tidak ada satu pun kasus
sistem salah mengenali satu karyawan sebagai karyawan lain. Revisi threshold dari 90 ke
82 persen tidak menambah risiko salah kenal orang sama sekali pada data ini, tapi secara
signifikan menurunkan tingkat verifikasi manual dari 40 persen menjadi 6.67 persen.
Dengan kata lain, penurunan threshold ini murni memperbaiki pengalaman pengguna
(karyawan lebih jarang diarahkan ke verifikasi manual), tanpa mengorbankan keamanan
identifikasi pada data yang diuji.

### 4.5 Employee similarity heatmap

Pasangan karyawan yang paling mirip satu sama lain, dari embedding rata-rata, adalah
ghani dan nadia dengan similarity 0.110. Angka ini masih jauh di bawah threshold
similarity yang dipakai untuk pencocokan (0.242), sehingga tidak berisiko tertukar satu
sama lain pada data saat ini. Nilai ini juga tidak berubah oleh revisi threshold
confidence, karena murni properti dari data wajah.

### 4.6 Ringkasan angka kunci (setelah revisi ke 82 persen)

| Metrik | Nilai |
|---|---|
| Jumlah karyawan diuji | 3 |
| Total embedding | 15, yaitu 5 foto dikali 3 orang |
| Rata-rata similarity genuine | 0.741 |
| Rata-rata similarity impostor | 0.076 |
| Equal Error Rate (EER) | 0.000 persen |
| Threshold similarity direkomendasikan | 0.242 |
| Confidence threshold (setelah revisi PIC) | 82 persen |
| Similarity dibutuhkan untuk confidence 82 persen | 0.700 |
| Akurasi simulasi leave-one-out | 93.33 persen |
| Tingkat verifikasi manual, simulasi | 6.67 persen |
| Tingkat salah kenal orang | 0 persen |

---

## 5. Kesimpulan Analisis

1. Model dan pipeline bekerja dengan benar. Separasi antara wajah yang sama dan wajah
   berbeda sangat jelas, 0.741 berbanding 0.076, dan tidak ada kasus salah kenal orang
   pada seluruh simulasi, baik di threshold 90 persen maupun 82 persen.
2. Revisi threshold dari 90 ke 82 persen terbukti tepat berdasarkan data. Akurasi
   auto-verifikasi naik dari 60 persen menjadi 93.33 persen, dan tingkat verifikasi
   manual turun dari 40 persen menjadi 6.67 persen, tanpa menambah risiko salah kenal
   orang sama sekali (tetap 0 persen pada kedua kondisi).
3. Sistem tetap bersifat konservatif, bukan ceroboh. Sisa 6.67 persen kasus yang
   diarahkan ke verifikasi manual bukan karena sistem ragu-ragu secara acak, tapi karena
   confidence-nya memang berada di bawah ambang batas yang ditetapkan.
4. Hasil ini adalah bukti konsep (PoC) yang valid, membuktikan pendekatan pretrained
   model ditambah kalibrasi berbasis data, tanpa training ulang, adalah keputusan yang
   tepat. Namun sample data (3 karyawan, 15 embedding) masih kecil dan belum
   representatif untuk seluruh populasi karyawan kantor, sehingga belum siap dipasang
   ke production tanpa langkah tambahan di bagian rekomendasi.

---

## 6. Keterbatasan yang Perlu Diketahui PIC

- Liveness dan anti-spoofing masih lemah. src/liveness.py saat ini hanya mengecek
  ukuran wajah dan ketajaman gambar (blur detection), bukan model anti-spoofing yang
  sesungguhnya. Ini belum bisa mencegah orang mencoba absen menggunakan foto di HP
  milik karyawan lain. Wajib diganti dengan model anti-spoofing khusus sebelum
  digunakan untuk absensi resmi.
- Sample data masih sangat kecil. Threshold dan confidence bound saat ini dihitung dari
  3 karyawan dan 15 embedding. Angka ini akan berubah, dan idealnya menjadi lebih
  stabil, setelah data diperluas.
- Belum diuji pada kondisi dunia nyata yang bervariasi, seperti pencahayaan berbeda,
  sudut kamera HP yang biasa dipakai untuk absen, penggunaan masker atau kacamata, atau
  perubahan wajah dari waktu ke waktu.
- Threshold 82 persen tervalidasi baik pada data 3 karyawan ini, tetapi belum tentu
  optimal begitu populasi karyawan bertambah dan variasi wajah semakin luas. Sebagian
  identitas baru berpotensi memiliki similarity genuine lebih rendah karena kualitas
  foto enrollment yang berbeda, sehingga kalibrasi ulang tetap diperlukan secara berkala.
- Skalabilitas pencarian. Saat ini pencocokan dilakukan dengan brute-force cosine
  similarity ke seluruh data. Ini masih cukup cepat untuk ratusan karyawan, tapi perlu
  diganti dengan index pencarian seperti FAISS jika jumlah karyawan bertambah banyak.
- Privasi data. Sistem hanya menyimpan embedding, yaitu vektor angka, bukan foto wajah
  mentah, tetapi embedding tetap termasuk data biometrik. Perlu dipastikan kebijakan
  consent dan retensi data sesuai aturan perusahaan sebelum deployment.

---

## 7. Rekomendasi Langkah Selanjutnya

Urutan prioritas yang disarankan sebelum sistem ini dipasang ke backend produksi:

1. Perluas data enrollment. Tambah jumlah karyawan uji, idealnya seluruh atau sebagian
   besar karyawan kantor, dan tambah jumlah foto per orang menjadi 8 sampai 10 foto,
   dengan variasi sudut dan pencahayaan, lalu jalankan ulang notebook untuk memastikan
   threshold 82 persen masih optimal pada populasi yang lebih besar.
2. Pantau tingkat verifikasi manual setelah data diperluas. Jika tingkat verifikasi
   manual naik signifikan di atas 6.67 persen seiring bertambahnya karyawan, evaluasi
   ulang apakah threshold 82 persen masih tepat atau perlu disesuaikan lagi.
3. Tambahkan model anti-spoofing dan liveness yang proper, menggantikan heuristik blur
   saat ini, sebelum sistem dipakai untuk absensi resmi.
4. Uji dengan kondisi dunia nyata, termasuk pencahayaan bervariasi, sudut kamera HP yang
   sebenarnya dipakai karyawan, dan kondisi seperti memakai masker atau kacamata.
5. Integrasi ke backend. Modul src/face_engine.py dan src/database.py sudah bersifat
   backend-ready dan tidak terikat ke framework tertentu. Contoh implementasi REST API
   tersedia di backend_api.py menggunakan FastAPI, menerima gambar dan mengembalikan
   employee_id, name, similarity, dan confidence_percent, siap dihubungkan ke aplikasi
   absensi kantor.
6. Pertimbangkan FAISS atau vector index sejenis apabila jumlah karyawan sudah
   berkembang cukup banyak, agar waktu pencocokan tetap cepat.
7. Pastikan kebijakan privasi data biometrik sudah dikoordinasikan dengan tim HR atau
   legal perusahaan sebelum data karyawan sungguhan digunakan secara luas.

---

## 8. Struktur Proyek

```
face-recognition-attendance/
├── app.py                          Aplikasi Streamlit: dashboard, enroll, verify, database, analytics
├── backend_api.py                  Contoh REST API (FastAPI) untuk integrasi backend
├── config.yaml                     Konfigurasi terpusat: model, path, threshold
├── requirements.txt
├── src/
│   ├── face_engine.py              Deteksi wajah, ekstraksi embedding, pencarian similarity
│   ├── database.py                 Penyimpanan embedding karyawan (JSON)
│   ├── liveness.py                 Cek liveness heuristik, masih placeholder
│   ├── attendance.py               Baca/tulis log absensi (CSV)
│   └── utils.py                    Load config, helper gambar, logging
├── notebooks/
│   └── face_recognition_pipeline.ipynb   Analisis lengkap: embedding, kalibrasi, visualisasi
├── data/
│   └── employees/<nama>/*.jpg      Foto enrollment, satu folder per karyawan
└── output/
    ├── models/                     Model ArcFace yang diunduh
    ├── embeddings/                 employee_embeddings.json, calibration.json, pipeline_summary.json
    ├── logs/                       attendance_log.csv
    └── visualizations/             Grafik hasil notebook
```

---

## 9. Cara Menjalankan

Membutuhkan Python 3.11, terpasang secara global tanpa virtual environment untuk PoC
ini.

```
pip install -r requirements.txt
```

Model buffalo_l akan otomatis terunduh saat pertama kali dijalankan, membutuhkan
koneksi internet sekali, setelah itu tersimpan secara lokal.

### 9.1 Jalankan notebook terlebih dahulu

Notebook adalah tahap analisis dan kalibrasi, wajib dijalankan sebelum memakai aplikasi.

```
jupyter notebook notebooks/face_recognition_pipeline.ipynb
```

Yang dilakukan notebook:

- Memuat foto karyawan dari data/employees. Jika belum ada foto asli, otomatis beralih
  ke mode data sintetis agar seluruh proses analisis tetap bisa dijalankan sebagai
  demonstrasi.
- Membangun distribusi similarity genuine vs impostor.
- Menghitung FAR, FRR, dan Equal Error Rate untuk berbagai nilai threshold.
- Merekomendasikan threshold similarity dan menyimpannya, beserta batas pemetaan
  confidence, ke output/embeddings/calibration.json.
- Memvisualisasikan ruang embedding menggunakan PCA dan t-SNE, serta heatmap similarity
  antar karyawan.
- Menjalankan simulasi pengujian end to end dengan confusion matrix.

Target confidence persen dapat diubah lewat variabel target_confidence_percent di
notebook, mengikuti nilai confidence_threshold_percent pada config.yaml. Setelah
mengubah target dan menjalankan ulang notebook, salin nilai
recommended_similarity_threshold, similarity_low_bound, dan similarity_high_bound dari
output/embeddings/calibration.json ke config.yaml.

### 9.2 Jalankan aplikasi Streamlit

```
streamlit run app.py
```

Halaman yang tersedia:

- Dashboard: ringkasan KPI, aktivitas absensi terbaru, breakdown status, tren harian.
- Verify Attendance: ambil foto lewat kamera, jalankan deteksi, embedding, cek
  liveness, dan pencarian similarity, tampilkan confidence gauge, dan minta konfirmasi
  nama yang teridentifikasi. Hasil di bawah threshold diarahkan ke verifikasi manual,
  dengan keterangan apakah penyebabnya confidence rendah atau liveness gagal.
- Enroll Employee: unggah beberapa foto, ekstrak dan simpan embedding.
- Employee Database: lihat, cari, dan hapus data karyawan yang terdaftar.
- Analytics: distribusi confidence score, jumlah check-in per karyawan, tren metode
  verifikasi dari waktu ke waktu, dan log lengkap.

Catatan: config.yaml membaca confidence_threshold_percent sekali saat aplikasi
dijalankan. Setelah mengubah nilai threshold di config.yaml, restart aplikasi Streamlit
agar perubahan terbaca.

### 9.3 Jalankan contoh backend API

```
uvicorn backend_api:app --reload
```

Endpoint utama adalah POST /verify, menerima file gambar, mengembalikan identitas,
similarity, dan confidence score dalam format JSON, siap dihubungkan ke aplikasi
absensi kantor.

---

## 10. Referensi

- Repository InsightFace: https://github.com/deepinsight/insightface
- Paper ArcFace: Deng, J., Guo, J., Xue, N., dan Zafeiriou, S. (2019). ArcFace:
  Additive Angular Margin Loss for Deep Face Recognition.
- File model yang digunakan: https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip