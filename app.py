import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Prediksi Diabetes - Pima Indians",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# 2. LOAD & PREPROCESS DATA (CACHED)
# ==========================================
@st.cache_data
def load_and_clean_data():
    # Load dataset lokal, jika gagal ambil dari URL publik
    try:
        df = pd.read_csv('diabetes.csv')
    except:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df = pd.read_csv(url)
    
    # Preprocessing: Mengatasi nilai 0 tidak logis dengan Median berdasarkan Outcome
    invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in invalid_zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))
    return df

df = load_and_clean_data()

# ==========================================
# 3. SPLIT DATA & TRAINING MODEL (CACHED)
# ==========================================
@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standard Scaling untuk KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Decision Tree Model
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    return knn, dt, scaler, X_test, X_test_scaled, y_test

knn_model, dt_model, scaler, X_test, X_test_scaled, y_test = train_models(df)

# ==========================================
# 4. SIDEBAR - INFORMASI PEMBUAT
# ==========================================
st.sidebar.title("🏥 Proyek UAS")
st.sidebar.subheader("Prediksi Risiko Diabetes")
st.sidebar.markdown("---")
st.sidebar.markdown("**Anggota Kelompok:**")
st.sidebar.write("1. [Annisa Salwa Harmiya] - [A11.2024.15962]")
st.sidebar.write("2. [Calysta Cayla Putri Anggraini] - [A11.2024.15978]")
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini dibuat menggunakan library Streamlit untuk memenuhi syarat UAS Pembelajaran Mesin.")

# ==========================================
# 5. HEADER UTAMA APLIKASI
# ==========================================
st.title("Aplikasi Deteksi Dini & Analisis Risiko Diabetes")
st.markdown("Aplikasi berbasis *Machine Learning* untuk mendeteksi risiko diabetes berdasarkan indikator medis klinis menggunakan *Pima Indians Diabetes Database*.")
st.markdown("---")

# ==========================================
# 6. MENU UTAMA MENGGUNAKAN TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dokumentasi Proyek", 
    "📊 Dashboard EDA Interaktif", 
    "🔬 Evaluasi Performa Model", 
    "🤖 Demo Prediksi Pasien", 
    "💡 Interpretasi & Insight Bisnis"
])

# ------------------------------------------
# TAB 1: DOKUMENTASI PROYEK
# ------------------------------------------
with tab1:
    st.header("1. Dokumentasi Dataset & Metodologi")
    st.write("""
    ### Deskripsi Masalah
    Diabetes melitus merupakan salah satu penyakit kronis paling mematikan yang ditandai dengan tingginya kadar glukosa darah. 
    Deteksi dini sangat krusial agar tindakan preventif dapat segera dilakukan untuk mencegah komplikasi lebih lanjut.
    
    ### Sumber Dataset
    Dataset yang digunakan dalam proyek ini adalah **Pima Indians Diabetes Database** dari **UCI Machine Learning Repository** (juga tersedia di Kaggle). 
    Dataset ini berisi data klinis dari pasien wanita keturunan suku Pima berumur minimal 21 tahun.
    """)
    
    st.markdown("**Struktur Data Awal (5 Data Teratas):**")
    st.dataframe(df.head())
    
    st.write("""
    ### Alur Metodologi (Pipeline):
    1. **Data Acquisition:** Memuat data Pima Indians Diabetes lokal.
    2. **Data Cleaning:** Mengganti nilai 0 tidak logis (pada Glucose, Blood Pressure, BMI, dll) menjadi nilai median dari masing-masing kelas target.
    3. **Feature Scaling:** Melakukan standardisasi skala fitur menggunakan `StandardScaler` khusus untuk algoritma KNN.
    4. **Modeling:** Melatih model klasifikasi **K-Nearest Neighbors (KNN)** dan **Decision Tree Classifier**.
    5. **Evaluation:** Membandingkan nilai Akurasi, Presisi, Recall, dan F1-Score pada kedua model untuk memilih model terbaik yang akan digunakan pada fitur demo prediksi.
    """)

# ------------------------------------------
# TAB 2: DASHBOARD EDA INTERAKTIF
# ------------------------------------------
with tab2:
    st.header("2. Dashboard Exploratory Data Analysis (EDA)")
    st.write("Eksplorasi dataset secara interaktif untuk memahami korelasi indikator medis dengan status diabetes pasien.")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Pengaturan Visualisasi")
        feature_to_plot = st.selectbox(
            "Pilih Fitur Medis untuk Dianalisis:",
            ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction']
        )
        st.markdown("""
        **Petunjuk Analisis:**
        * Perhatikan pergeseran boxplot antara pasien Sehat (0) dan Diabetes (1).
        * Kadar glukosa tinggi dan BMI tinggi secara umum menunjukkan korelasi yang sangat kuat dengan risiko positif diabetes.
        """)
        
    with col_right:
        # Visualisasi Boxplot berdasarkan input user
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Outcome', y=feature_to_plot, data=df, palette='Set1', ax=ax)
        ax.set_title(f"Distribusi Fitur '{feature_to_plot}' Berdasarkan Outcome (0=Sehat, 1=Diabetes)")
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Korelasi Antar Fitur Medis (Correlation Heatmap)")
    
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)

# ------------------------------------------
# TAB 3: EVALUASI PERFORMA MODEL
# ------------------------------------------
with tab3:
    st.header("3. Perbandingan & Evaluasi Performa Model")
    st.write("Di bawah ini adalah perbandingan kinerja antara algoritma **K-Nearest Neighbors (KNN)** dan **Decision Tree Classifier**.")
    
    # Hitung prediksi data uji untuk evaluasi
    y_pred_knn = knn_model.predict(X_test_scaled)
    y_pred_dt = dt_model.predict(X_test)
    
    # Ringkasan Metrik
    perf_metrics = {
        "Metrik Evaluasi": ["Accuracy (Akurasi)", "Precision (Presisi)", "Recall (Sensitivitas)", "F1-Score"],
        "K-Nearest Neighbors (KNN)": [
            f"{accuracy_score(y_test, y_pred_knn):.2%}",
            f"{precision_score(y_test, y_pred_knn):.2%}",
            f"{recall_score(y_test, y_pred_knn):.2%}",
            f"{f1_score(y_test, y_pred_knn):.2%}"
        ],
        "Decision Tree Classifier": [
            f"{accuracy_score(y_test, y_pred_dt):.2%}",
            f"{precision_score(y_test, y_pred_dt):.2%}",
            f"{recall_score(y_test, y_pred_dt):.2%}",
            f"{f1_score(y_test, y_pred_dt):.2%}"
        ]
    }
    st.table(pd.DataFrame(perf_metrics))
    
    # Visualisasi Confusion Matrix berdampingan
    st.subheader("Visualisasi Confusion Matrix")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confusion Matrix: KNN**")
        fig_knn_cm, ax_knn_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=ax_knn_cm)
        ax_knn_cm.set_xlabel('Prediksi')
        ax_knn_cm.set_ylabel('Sebenarnya')
        st.pyplot(fig_knn_cm)
        
    with col2:
        st.write("**Confusion Matrix: Decision Tree**")
        fig_dt_cm, ax_dt_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Oranges', ax=ax_dt_cm)
        ax_dt_cm.set_xlabel('Prediksi')
        ax_dt_cm.set_ylabel('Sebenarnya')
        st.pyplot(fig_dt_cm)

# ------------------------------------------
# TAB 4: DEMO PREDIKSI PASIEN
# ------------------------------------------
with tab4:
    st.header("4. Form Deteksi Risiko Diabetes Pasien")
    st.write("Silakan masukkan parameter klinis pasien di bawah ini untuk mendapatkan hasil deteksi langsung dari model Machine Learning terbaik (KNN).")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preg = st.number_input("Kehamilan (Pregnancies):", min_value=0, max_value=20, value=2, step=1)
        glucose = st.number_input("Kadar Glukosa Darah (Glucose):", min_value=40.0, max_value=250.0, value=117.0)
        bp = st.number_input("Tekanan Darah Diastolik (Blood Pressure):", min_value=30.0, max_value=150.0, value=72.0)
        
    with col2:
        skin = st.number_input("Tebal Lipatan Kulit (Skin Thickness - mm):", min_value=5.0, max_value=100.0, value=23.0)
        insulin = st.number_input("Kadar Insulin (Insulin - mu U/ml):", min_value=10.0, max_value=900.0, value=30.0)
        bmi = st.number_input("Indeks Massa Tubuh (BMI):", min_value=10.0, max_value=70.0, value=32.0)
        
    with col3:
        dpf = st.number_input("Fungsi Silsilah Diabetes (Diabetes Pedigree Function):", min_value=0.05, max_value=3.0, value=0.37, format="%.3f")
        age = st.number_input("Usia Pasien (Tahun):", min_value=21, max_value=100, value=29, step=1)
    
    # Tombol Aksi Prediksi
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔥 DETEKSI RISIKO DIABETES", use_container_width=True):
        # 1. Satukan input menjadi array numpy
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        
        # 2. Lakukan penskalaan data menggunakan StandardScaler terlatih
        input_data_scaled = scaler.transform(input_data)
        
        # 3. Prediksi menggunakan model KNN
        prediction = knn_model.predict(input_data_scaled)
        prediction_proba = knn_model.predict_proba(input_data_scaled)[0][1]
        
        # 4. Tampilkan Hasil
        st.markdown("---")
        st.subheader("Hasil Diagnosis Medis:")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **STATUS: BERISIKO TINGGI DIABETES (Kategori Positif)**")
            st.write(f"Pasien memiliki probabilitas kecenderungan risiko diabetes sebesar **{prediction_proba:.2%}** berdasarkan pola rekam medis.")
            st.warning("Rekomendasi: Sangat disarankan bagi pasien untuk melakukan pemeriksaan laboratorium formal (HbA1c/Tes Toleransi Glukosa) dan berkonsultasi dengan Dokter Spesialis Penyakit Dalam.")
        else:
            st.success(f"✅ **STATUS: RISIKO RENDAH / SEHAT (Kategori Negatif)**")
            st.write(f"Pasien memiliki probabilitas risiko terkena diabetes yang sangat minim, yaitu sebesar **{prediction_proba:.2%}**.")
            st.info("Rekomendasi: Tetap pertahankan pola makan sehat, rutin berolahraga, serta lakukan pemeriksaan berkala untuk menjaga stabilitas gula darah.")

# ------------------------------------------
# TAB 5: INTERPRETASI & INSIGHT BISNIS
# ------------------------------------------
with tab5:
    st.header("5. Interpretasi Model & Insight Bisnis")
    st.markdown("""
    ### Analisis Hasil & Insight Bisnis untuk Manajemen Kesehatan:
    
    1. **Glukosa sebagai Prediktor Utama:**
       Kadar glukosa darah 2 jam setelah tes toleransi oral merupakan fitur dengan korelasi tertinggi terhadap diabetes. Dalam perspektif bisnis/operasional rumah sakit, deteksi dini berbasis glukosa harus diprioritaskan lewat program penyaringan massal (screening).
    
    2. **Justifikasi Pemilihan Model (KNN):**
       Model **K-Nearest Neighbors (KNN)** dipilih sebagai model utama di dashboard demo karena memberikan performa yang lebih seimbang antara tingkat akurasi tinggi dan sensitivitas (Recall) yang baik dalam mengidentifikasi pasien sakit tanpa menghasilkan banyak kesalahan prediksi negatif palsu (False Negative).
       
    3. **Rekomendasi Strategis Manajemen Medis:**
       * **Automated Screening:** Mengintegrasikan sistem prediksi berbasis ML ini ke rekam medis elektronik (EHR) rumah sakit agar secara otomatis memberi tanda peringatan (alert) bagi pasien berisiko tinggi saat melakukan medical checkup rutin.
       * **Pencegahan Berbasis Segmentasi BMI & Umur:** Membuat kampanye gaya hidup sehat terarah bagi populasi wanita di atas 30 tahun yang memiliki indeks BMI tinggi guna menekan laju pertumbuhan pengidap diabetes melitus di lingkungan masyarakat.
    """)