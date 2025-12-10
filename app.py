import streamlit as st
import pandas as pd
import joblib

# Load Model dan kelengkapannya
model = joblib.load('knn_full_model.pkl')
scaler = joblib.load('scaler_full.pkl')
le = joblib.load('label_encoder_full.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")
st.markdown("Menggunakan **K-Nearest Neighbors (KNN)** dengan **34 Fitur Lengkap**")
st.info("Masukkan data mahasiswa secara lengkap di bawah ini untuk melakukan prediksi.")

# Form Input
with st.form("prediction_form"):
    # Kita bagi 34 kolom menjadi kategori agar GUI rapi
    # Tabulasi Input
    tab1, tab2, tab3 = st.tabs(["👤 Data Demografi", "💰 Sosial & Ekonomi", "📚 Data Akademik"])

    # Dictionary untuk menampung input user
    input_data = {}

    with tab1:
        st.header("Data Pribadi & Keluarga")
        col1, col2 = st.columns(2)
        with col1:
            input_data['Marital status'] = st.selectbox("Status Pernikahan (1=Single, 2=Married, etc)", [1,2,3,4,5,6])
            input_data['Gender'] = st.selectbox("Jenis Kelamin (1=Laki, 0=Perempuan)", [1, 0])
            input_data['Age at enrollment'] = st.number_input("Umur saat Masuk", 17, 70, 20)
            input_data['Nacionality'] = st.selectbox("Kewarganegaraan (Kode)", [1, 41, 2, 6, 11, 13, 14, 17, 101])
            input_data['International'] = st.selectbox("Mahasiswa Internasional?", [0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        
        with col2:
            input_data["Mother's qualification"] = st.number_input("Pendidikan Ibu (Kode)", 1, 44, 1)
            input_data["Father's qualification"] = st.number_input("Pendidikan Ayah (Kode)", 1, 44, 1)
            input_data["Mother's occupation"] = st.number_input("Pekerjaan Ibu (Kode)", 1, 44, 1)
            input_data["Father's occupation"] = st.number_input("Pekerjaan Ayah (Kode)", 1, 44, 1)
            input_data['Displaced'] = st.selectbox("Displaced (Perantau?)", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
            input_data['Educational special needs'] = st.selectbox("Kebutuhan Khusus?", [0, 1])

    with tab2:
        st.header("Faktor Sosial Ekonomi")
        col3, col4 = st.columns(2)
        with col3:
            input_data['Debtor'] = st.selectbox("Memiliki Hutang?", [0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")
            input_data['Tuition fees up to date'] = st.selectbox("SPP Lancar?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
            input_data['Scholarship holder'] = st.selectbox("Penerima Beasiswa?", [0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        
        with col4:
            input_data['Unemployment rate'] = st.number_input("Tingkat Pengangguran Negara (%)", 0.0, 20.0, 10.0)
            input_data['Inflation rate'] = st.number_input("Tingkat Inflasi (%)", -5.0, 20.0, 1.4)
            input_data['GDP'] = st.number_input("GDP", -10.0, 10.0, 0.0)
            input_data['Application mode'] = st.number_input("Mode Aplikasi (Kode)", 1, 18, 1)
            input_data['Application order'] = st.number_input("Urutan Pilihan", 0, 9, 1)
            input_data['Course'] = st.number_input("Kode Jurusan", 1, 9999, 33)
            input_data['Daytime/evening attendance'] = st.selectbox("Waktu Kuliah", [1, 0], format_func=lambda x: "Siang" if x==1 else "Malam")
            input_data['Previous qualification'] = st.number_input("Kualifikasi Sebelumnya", 1, 43, 1)

    with tab3:
        st.header("Performa Akademik (Semester 1 & 2)")
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Semester 1")
            input_data['Curricular units 1st sem (credited)'] = st.number_input("Sem 1: SKS Diakui", 0, 20, 0)
            input_data['Curricular units 1st sem (enrolled)'] = st.number_input("Sem 1: SKS Diambil", 0, 20, 5)
            input_data['Curricular units 1st sem (evaluations)'] = st.number_input("Sem 1: Jumlah Evaluasi", 0, 20, 5)
            input_data['Curricular units 1st sem (approved)'] = st.number_input("Sem 1: SKS Lulus", 0, 20, 5)
            input_data['Curricular units 1st sem (grade)'] = st.number_input("Sem 1: Rata-rata Nilai", 0.0, 20.0, 12.0)
            input_data['Curricular units 1st sem (without evaluations)'] = st.number_input("Sem 1: Tanpa Evaluasi", 0, 10, 0)

        with col6:
            st.subheader("Semester 2")
            input_data['Curricular units 2nd sem (credited)'] = st.number_input("Sem 2: SKS Diakui", 0, 20, 0)
            input_data['Curricular units 2nd sem (enrolled)'] = st.number_input("Sem 2: SKS Diambil", 0, 20, 5)
            input_data['Curricular units 2nd sem (evaluations)'] = st.number_input("Sem 2: Jumlah Evaluasi", 0, 20, 5)
            input_data['Curricular units 2nd sem (approved)'] = st.number_input("Sem 2: SKS Lulus", 0, 20, 5)
            input_data['Curricular units 2nd sem (grade)'] = st.number_input("Sem 2: Rata-rata Nilai", 0.0, 20.0, 12.0)
            input_data['Curricular units 2nd sem (without evaluations)'] = st.number_input("Sem 2: Tanpa Evaluasi", 0, 10, 0)

    # Tombol Submit
    submit = st.form_submit_button("🚀 Prediksi Sekarang")

if submit:
    # 1. Konversi input dictionary ke DataFrame sesuai urutan feature_names
    # Ini menjamin urutan kolom SAMA PERSIS dengan saat training
    input_df = pd.DataFrame([input_data])
    
    # Reorder kolom untuk keamanan (jika urutan di dict berantakan)
    input_df = input_df[feature_names]

    # 2. Scaling Data (Wajib untuk KNN)
    input_scaled = scaler.transform(input_df)

    # 3. Prediksi
    prediction_idx = model.predict(input_scaled)
    prediction_label = le.inverse_transform(prediction_idx)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # 4. Tampilkan Hasil
    st.divider()
    st.subheader("Hasil Prediksi")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if prediction_label == 'Dropout':
            st.error(f"Status: **{prediction_label}**")
        elif prediction_label == 'Graduate':
            st.success(f"Status: **{prediction_label}**")
        else:
            st.warning(f"Status: **{prediction_label}**")
            
    with col_res2:
        st.write("Probabilitas Model:")
        classes = le.classes_
        for i, class_name in enumerate(classes):
            st.progress(float(probabilities[i]), text=f"{class_name}: {probabilities[i]*100:.1f}%")