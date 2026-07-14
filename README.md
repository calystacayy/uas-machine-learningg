\# **Dokumentasi Projek Skrining Awal \& Evaluasi Risiko Diabetes Melitus**



Projek ini dirancang dan dikembangkan guna memenuhi syarat penilaian Ujian Akhir Semester (UAS) pada mata kuliah Pembelajaran Mesin. Fokus utama dari riset berskala kecil ini adalah mengintegrasikan analisis data medis, pemodelan klasifikasi cerdas (\*Machine Learning\*), dan implementasi antarmuka berbasis web agar proses deteksi dini penyakit diabetes dapat diakses secara praktis.



\---



\# Tim Penyusun (Kelas: A11.4407)

\- Annisa Salwa Harmiya – NIM: A11.2O24.15962

\- Calysta Cayla Putri Anggraini – NIM: A11.2024.15978



**Dosen Pengampu**: Prof. Ir. Heru Agus Santoso, Ph.D, IPM, ASEAN Eng.  

**Program Studi**: Teknik Informatika - S1  

Fakultas Ilmu Komputer, Universitas Dian Nuswantoro



\---



\# Akses Cepat Projek

\- Tautan Aplikasi Ter-deploy (Streamlit): https://uas-machine-learning-annisa-calysta.streamlit.app/

\- Tautan Repositori Projek (GitHub): https://github.com/calysstaa/uas-machine-learning



\---



\# Struktur Berkas dalam Repositori

Berikut adalah susunan file projek yang tersimpan di dalam repositori ini agar terlihat rapi dan terstruktur:



uas-machine-learning/

│

├── diabetes.csv          			          # File dataset mentah Pima Indians Diabetes

├── app.py                			          # Skrip utama aplikasi web interaktif Streamlit

├── notebook\_prediksi\_diabetes.ipynb		# Dokumen kerja proses EDA, Preprocessing, dan Modeling (Jupyter)

├── README.md             			          # File panduan dan informasi projek (Dokumen Ini)

└── requirements.txt                      # Berisi daftar pustaka (packages) luar yang digunakan dalam projek ini. Berkas ini dibaca otomatis oleh server                                                      Streamlit Cloud saat proses deployment untuk memasang lingkungan kerja secara instan.



\# Ikhtisar Metodologi \& Pipeline Pemodelan

Projek ini memanfaatkan Pima Indians Diabetes Database yang bersumber dari UCI Machine Learning Repository, mencakup data rekam medis dari 768 pasien wanita. Langkah-langkah pengembangan sistem yang kami lakukan meliputi:



* Pembersihan Data (Data Cleaning): Menangani anomali medis berupa nilai 0 yang tidak masuk akal pada fitur vital (Glukosa, Tekanan Darah, Insulin, Ketebalan Kulit, dan BMI). Nilai janggal ini kami konversi menjadi NaN lalu diimputasi menggunakan nilai median spesifik berdasarkan label target pasien (Outcome).



* Standardisasi Nilai (Feature Scaling): Menerapkan fungsi StandardScaler untuk menyamakan rentang nilai fitur numerik. Langkah ini wajib dilakukan agar komputasi berbasis jarak pada model KNN tidak didominasi oleh variabel dengan skala angka yang besar.



* Eksperimen Algoritma: Melatih dan menguji dua rumpun model klasifikasi yang berbeda, yaitu K-Nearest Neighbors (KNN) dan Decision Tree Classifier, dengan skema pembagian data 80% untuk pelatihan (train) dan 20% untuk pengujian (test).



* Pemilihan Model Terbaik: Berdasarkan hasil pengujian independen, algoritma KNN dipilih sebagai model akhir untuk sistem produksi. Model ini mencatatkan tingkat akurasi tertinggi sebesar 82.47% serta skor Recall sebesar 71.43%, yang dinilai sangat aman untuk meminimalkan risiko luputnya pasien sakit dalam diagnosis medis.



\# Panduan Menjalankan Aplikasi di Lingkungan Lokal

Bagi yang ingin menguji dan mengaktifkan aplikasi web ini di perangkat komputer/laptop pribadi secara lokal, silakan ikuti petunjuk berikut:



1\. Persiapan Awal

Pastikan perangkat Anda sudah terpasang Python versi 3.8 ke atas atau menggunakan lingkungan Anaconda.



2\. Mengunduh Projek

Buka Terminal atau Command Prompt (CMD) Anda, lalu jalankan perintah kloning berikut:



Bash

git clone \[https://github.com/] https://github.com/calysstaa/uas-machine-learning

cd uas-machine-learning

3\. Memasang Library Pendukung

Instal seluruh package Python yang dibutuhkan agar program tidak mengalami error saat dijalankan:



Bash

pip install streamlit pandas numpy matplotlib seaborn scikit-learn

4\. Menyalakan Server Streamlit

Eksekusi perintah di bawah ini untuk menghidupkan aplikasi web lokal:



Bash

streamlit run app.py

Tunggu beberapa detik, browser internet Anda akan langsung terbuka otomatis dan memuat dasbor interaktif pada alamat lokal http://localhost:8501.



\# Cakupan Fitur Aplikasi Web

Aplikasi ini didesain user-friendly dengan menggunakan sistem navigasi multi-tab yang memuat komponen wajib:



Tab 1: Dokumentasi Proyek – Informasi dasar riset, latar belakang medis, beserta alur pipa kerja (pipeline).



Tab 2: Dashboard EDA Interaktif – Fitur dropdown dinamis untuk memunculkan grafik boxplot dan korelasi indikator kesehatan secara real-time.



Tab 3: Evaluasi Performa Model – Rekap tabel komparasi performa metrik beserta visualisasi Confusion Matrix berdampingan.



Tab 4: Demo Prediksi Pasien – Lembar form digital untuk menginput rekam medis pasien baru guna mendapatkan hasil diagnosis instan berlabel warna (Merah = Risiko Tinggi, Hijau = Risiko Rendah).



Tab 5: Interpretasi \& Insight Bisnis – Rekomendasi strategis dan nilai konkrit implementasi sistem bagi pihak manajemen rumah sakit.

