import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="wide")

# --- LOAD MODEL ---
try:
    artifacts = joblib.load('student_prediction_artifacts.pkl')
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    target_le = artifacts['target_encoder']
    feature_names = artifacts['feature_names']
    cat_cols = artifacts['cat_cols']
except FileNotFoundError:
    st.error("Error: File model tidak ditemukan! Harap jalankan 'python train_model.py' dulu.")
    st.stop()

# --- CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Sistem Prediksi Kelulusan Mahasiswa")
st.markdown("Masukkan data akademik dan sosial ekonomi untuk menganalisis potensi kelulusan.")
st.divider()

# --- HELPER FUNCTION ---
def get_scale_value(label):
    mapping = {
        "Sangat Rendah (1)": 1, "Rendah (2)": 2, "Sedang (3)": 3, "Tinggi (4)": 4, "Sangat Tinggi (5)": 5,
        "Sangat Santai (1)": 1, "Santai (2)": 2, "Cukup Tertekan (3)": 3, "Stres (4)": 4, "Sangat Stres (5)": 5
    }
    return mapping[label]

with st.form("input_form"):
    # TAB MENU
    tab1, tab2, tab3, tab4 = st.tabs(["Data Diri", "Ekonomi", "Akademik", "Psikologis & Aktivitas"])
    input_data = {}

    # TAB 1: DATA DIRI
    with tab1:
        st.subheader("Informasi Dasar")
        c1, c2 = st.columns(2)
        with c1:
            input_data['UsiaMasuk'] = st.number_input("Usia Masuk Kuliah", 15, 50, 18)
            input_data['JalurMasuk'] = st.selectbox("Jalur Masuk", encoders['JalurMasuk'].classes_)
        with c2:
            input_data['PendidikanSebelumnya'] = st.selectbox("Asal Sekolah", encoders['PendidikanSebelumnya'].classes_)
            input_data['Jarak_km'] = st.number_input("Jarak Tempat Tinggal (km)", 0.0, 100.0, 5.0)
            input_data['Transportasi'] = st.selectbox("Transportasi Utama", encoders['Transportasi'].classes_)

    # TAB 2: EKONOMI
    with tab2:
        st.subheader("Latar Belakang Keluarga")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Pekerjaan & Pendidikan Orang Tua**")
            input_data['PendidikanIbu'] = st.selectbox("Pend. Terakhir Ibu", encoders['PendidikanIbu'].classes_)
            input_data['PendidikanAyah'] = st.selectbox("Pend. Terakhir Ayah", encoders['PendidikanAyah'].classes_)
            input_data['PekerjaanIbu'] = st.selectbox("Pekerjaan Ibu", encoders['PekerjaanIbu'].classes_)
            input_data['PekerjaanAyah'] = st.selectbox("Pekerjaan Ayah", encoders['PekerjaanAyah'].classes_)
        with c4:
            st.markdown("**Kondisi Finansial**")
            input_data['PendapatanKeluarga_Juta'] = st.number_input("Total Gaji Ortu (Juta Rupiah)", 0.0, 100.0, 5.0, step=0.5)
            input_data['JenisTempatTinggal'] = st.selectbox("Status Tempat Tinggal", encoders['JenisTempatTinggal'].classes_)
            input_data['KesulitanEkonomi'] = st.selectbox("Kondisi Ekonomi", [0, 1], format_func=lambda x: "Stabil / Cukup" if x==0 else "Ada Kesulitan Ekonomi")
            input_data['PenerimaBeasiswa'] = st.selectbox("Status Beasiswa", [0, 1], format_func=lambda x: "Bukan Penerima" if x==0 else "Penerima Beasiswa")

    # TAB 3: AKADEMIK
    with tab3:
        st.subheader("Performa Akademik (Tahun Pertama)")
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**:blue[Semester 1]**")
            input_data['SKS_Diambil_S1'] = st.number_input("SKS Diambil (Sems 1)", 0, 24, 20)
            input_data['SKS_Lulus_S1'] = st.number_input("SKS Lulus (Sems 1)", 0, 24, 20)
            input_data['IP_S1'] = st.number_input("IP Semester 1", 0.00, 4.00, 3.50)
            input_data['NilaiRata_S1'] = st.number_input("Rata-rata Nilai Angka Sems 1 (0-100)", 0.0, 100.0, 80.0)
            input_data['Presensi_S1'] = st.slider("Kehadiran Kuliah Sems 1 (%)", 0, 100, 90)
        with c6:
            st.markdown("**:blue[Semester 2]**")
            input_data['SKS_Diambil_S2'] = st.number_input("SKS Diambil (Sems 2)", 0, 24, 20)
            input_data['SKS_Lulus_S2'] = st.number_input("SKS Lulus (Sems 2)", 0, 24, 20)
            input_data['IP_S2'] = st.number_input("IP Semester 2", 0.00, 4.00, 3.50)
            input_data['NilaiRata_S2'] = st.number_input("Rata-rata Nilai Angka Sems 2 (0-100)", 0.0, 100.0, 80.0)
            input_data['Presensi_S2'] = st.slider("Kehadiran Kuliah Sems 2 (%)", 0, 100, 90)
        
        st.markdown("---")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            input_data['UKT_TepatWaktu'] = st.radio("Riwayat Pembayaran UKT", [1, 0], format_func=lambda x: "Selalu Tepat Waktu" if x==1 else "Pernah Menunggak")
        with col_stat2:
            input_data['PeringatanAkademik'] = st.radio("Status Surat Peringatan (SP)", [0, 1], format_func=lambda x: "Aman (Tidak Ada)" if x==0 else "Pernah Mendapat SP")

    # TAB 4: PSIKOLOGIS & AKTIVITAS
    with tab4:
        st.subheader("Faktor Pendukung")
        c7, c8 = st.columns(2)
        with c7:
            st.markdown("**Keaktifan & Prestasi**")
            input_data['KeikutsertaanKlub'] = st.selectbox("Keikutsertaan Organisasi", [0, 1], format_func=lambda x: "Tidak Ikut" if x==0 else "Aktif Mengikuti")
            input_data['PeranOrganisasi'] = st.selectbox("Peran Dominan", encoders['PeranOrganisasi'].classes_)
            input_data['KeikutsertaanLomba'] = st.selectbox("Riwayat Lomba", [0, 1], format_func=lambda x: "Tidak Pernah" if x==0 else "Pernah Mengikuti")
            input_data['Pencapaian'] = st.slider("Jumlah Sertifikat/Prestasi", 0, 10, 0)

        with c8:
            st.markdown("**Kondisi Mental & Dukungan**")
            motivasi_str = st.select_slider("Tingkat Motivasi Belajar", options=["Sangat Rendah (1)", "Rendah (2)", "Sedang (3)", "Tinggi (4)", "Sangat Tinggi (5)"], value="Tinggi (4)")
            input_data['MotivasiBelajar'] = get_scale_value(motivasi_str)
            
            dukungan_str = st.select_slider("Dukungan Orang Tua", options=["Sangat Rendah (1)", "Rendah (2)", "Sedang (3)", "Tinggi (4)", "Sangat Tinggi (5)"], value="Sangat Tinggi (5)")
            input_data['DukunganOrangTua'] = get_scale_value(dukungan_str)
            
            stres_str = st.select_slider("Tingkat Stres Mahasiswa", options=["Sangat Santai (1)", "Santai (2)", "Cukup Tertekan (3)", "Stres (4)", "Sangat Stres (5)"], value="Santai (2)")
            input_data['TingkatStres'] = get_scale_value(stres_str)

        # Default features (Hidden)
        input_data['AnakPertamaKuliah'] = 0
        input_data['MahasiswaBekerja'] = 0
        input_data['PunyaLaptop'] = 1
        input_data['SertifOspekFakultas'] = 1
        input_data['SertifOspekDepartemen'] = 1

    st.markdown("---")
    submit = st.form_submit_button("Mulai Proses Analisis", type="primary", use_container_width=True)

# --- LOGIKA PREDIKSI ---
if submit:
    # SIAPKAN DATA
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]
    
    # ENCODE & SCALE
    for col in cat_cols:
        le = encoders[col]
        df_input[col] = le.transform(df_input[col])
    
    X_input = scaler.transform(df_input)
    
    # HITUNG PREDIKSI & PROBABILITAS
    pred_idx = model.predict(X_input)[0]
    pred_label = target_le.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(X_input)[0]
    
    # Ambil Nilai Persentase Tertinggi
    max_prob = np.max(proba) * 100
    
    # TAMPILKAN HASIL
    st.divider()
    st.subheader("Hasil Analisis Sistem")
    
    col_kiri, col_kanan = st.columns([1, 2])
    
    with col_kiri:
        # Menampilkan teks formal dengan persentase
        if pred_label == 'Dropout':
            st.error(f"Prediksi: {pred_label}")
            st.write(f"**Tingkat Keyakinan Sistem: {max_prob:.2f}%**")
            st.markdown("---")
            st.markdown("**Rekomendasi:** Mahasiswa terdeteksi berisiko tinggi. Perlu pendampingan akademik intensif segera.")
            
        elif pred_label == 'Lulus Terlambat':
            st.warning(f"Prediksi: {pred_label}")
            st.write(f"**Tingkat Keyakinan Sistem: {max_prob:.2f}%**")
            st.markdown("---")
            st.markdown("**Rekomendasi:** Mahasiswa perlu evaluasi beban studi agar bisa mengejar ketertinggalan.")
            
        else:
            st.success(f"Prediksi: {pred_label}")
            st.write(f"**Tingkat Keyakinan Sistem: {max_prob:.2f}%**")
            st.markdown("---")
            st.markdown("**Keterangan:** Performa mahasiswa terpantau baik dan sesuai jalur (On-Track).")
            
    with col_kanan:
        st.write("Detail Probabilitas per Kategori:")
        # Chart sederhana
        prob_df = pd.DataFrame({
            "Status": target_le.classes_,
            "Probabilitas (%)": proba * 100 
        }).set_index("Status")
        
        # Format angka di tabel chart 
        st.bar_chart(prob_df)