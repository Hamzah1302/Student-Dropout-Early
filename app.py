import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut – Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26,35,126,0.3);
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 800; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.85; font-size: 1rem; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #3949ab;
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #1a237e; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .result-dropout  { background:#fdecea; border:2px solid #e53935; border-radius:12px; padding:1.5rem; text-align:center; }
    .result-enrolled { background:#fff8e1; border:2px solid #f9a825; border-radius:12px; padding:1.5rem; text-align:center; }
    .result-graduate { background:#e8f5e9; border:2px solid #2e7d32; border-radius:12px; padding:1.5rem; text-align:center; }
    .result-dropout  h2 { color:#c62828; }
    .result-enrolled h2 { color:#f57f17; }
    .result-graduate h2 { color:#1b5e20; }
    .stSelectbox label, .stSlider label, .stNumberInput label { font-weight: 600; color: #1a237e; }
    .section-title { color:#1a237e; font-weight:700; font-size:1.1rem;
                     border-bottom:2px solid #e8eaf6; padding-bottom:0.4rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────────────────────────
# Notebook terbaru (v3) menghasilkan 4 file:
#   best_model.pkl    → Pipeline lengkap (ColumnTransformer + XGBoost)
#   label_encoder.pkl → LabelEncoder untuk kolom Status
#   feature_names.pkl → Daftar 36 kolom RAW input (sebelum feature engineering)
#   model_info.pkl    → Metadata model (nama, metrik, daftar fitur)
#
# CATATAN: scaler.pkl TIDAK ADA lagi karena scaling sudah di dalam Pipeline.
# Feature engineering (approval_rate, dll) dilakukan di fungsi predict di bawah.

@st.cache_resource
def load_model():
    try:
        model  = joblib.load('model/best_model.pkl')
        le     = joblib.load('model/label_encoder.pkl')
        feats  = joblib.load('model/feature_names.pkl')
        info   = joblib.load('model/model_info.pkl')
        return model, le, feats, info
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        return None, None, None, None

model, le, feature_names, model_info = load_model()

# ── Daftar 36 kolom RAW (input dari user, sebelum feature engineering) ─────────
RAW_COLS = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality',
    'Mothers_qualification', 'Fathers_qualification',
    'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs',
    'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan outlier clipping dan feature engineering yang sama
    seperti di notebook v3 sebelum prediksi.
    Input : DataFrame dengan 36 kolom RAW
    Output: DataFrame dengan 42 kolom (36 raw + 6 engineered)
    """
    df = df.copy()

    # 1. Outlier handling (winsorizing — batas dari training data)
    clip_bounds = {
        'Age_at_enrollment':                     (18.0, 50.0),
        'Curricular_units_1st_sem_evaluations':  (0.0,  21.0),
        'Curricular_units_2nd_sem_evaluations':  (0.0,  19.0),
        'Admission_grade':                       (99.0, 166.8),
        'Previous_qualification_grade':          (100.0, 170.0),
    }
    for col, (lo, hi) in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 2. Feature engineering (sama persis dengan notebook v3)
    df['approval_rate_1st'] = df['Curricular_units_1st_sem_approved'] / \
                              (df['Curricular_units_1st_sem_enrolled'] + 1e-6)
    df['approval_rate_2nd'] = df['Curricular_units_2nd_sem_approved'] / \
                              (df['Curricular_units_2nd_sem_enrolled'] + 1e-6)
    df['total_approved']    = df['Curricular_units_1st_sem_approved'] + \
                              df['Curricular_units_2nd_sem_approved']
    df['avg_grade']         = (df['Curricular_units_1st_sem_grade'] + \
                               df['Curricular_units_2nd_sem_grade']) / 2
    df['financial_risk']    = ((1 - df['Tuition_fees_up_to_date']) + \
                               df['Debtor']).clip(0, 2)
    df['enrolled_total']    = df['Curricular_units_1st_sem_enrolled'] + \
                              df['Curricular_units_2nd_sem_enrolled']
    return df

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Jaya Jaya Institut</h1>
    <p>Student Dropout Early Warning System • Sistem Deteksi Dini Risiko Dropout Mahasiswa</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(" File model tidak ditemukan. Pastikan folder `model/` berisi file hasil training notebook.")
    st.info("Jalankan notebook_v3.ipynb terlebih dahulu untuk menghasilkan: best_model.pkl, label_encoder.pkl, feature_names.pkl, model_info.pkl")
    st.stop()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini membantu **Jaya Jaya Institut** mendeteksi secara dini mahasiswa
    yang berpotensi **dropout**, sehingga dapat diberikan bimbingan khusus tepat waktu.

    **Model:** {}
    **Akurasi:** {:.1f}%
    **F1-Weighted:** {:.1f}%
    **ROC-AUC:** {:.3f}
    """.format(
        model_info.get('model_name', '-'),
        model_info.get('accuracy', 0) * 100,
        model_info.get('f1_weighted', 0) * 100,
        model_info.get('roc_auc', 0),
    ))

    st.divider()
    st.markdown("### Label Prediksi")
    st.markdown("""
    - 🔴 **Dropout** — Berisiko tidak menyelesaikan studi
    - 🟡 **Enrolled** — Masih aktif berkuliah
    - 🟢 **Graduate** — Diprediksi lulus tepat waktu
    """)

    st.divider()
    st.markdown("### Info Pipeline")
    st.markdown("""
    Model ini menggunakan **Pipeline** terintegrasi:
    1. `ColumnTransformer` — StandardScaler + OrdinalEncoder
    2. `XGBoost Classifier`

    Input cukup **36 kolom raw** — feature engineering & scaling
    dilakukan otomatis di dalam pipeline.
    """)

# ── Main Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Prediksi Individu", "Prediksi Batch (CSV)"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Prediksi Individu
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Masukkan Data Mahasiswa")
    st.caption("Isi formulir berikut untuk memprediksi status mahasiswa secara individual.")

    with st.form("prediction_form"):

        # ── Informasi Pribadi ──────────────────────────────────────────────────
        st.markdown('<p class="section-title"> Informasi Pribadi & Demografis</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            marital_status = st.selectbox("Status Pernikahan",
                options=[1,2,3,4,5,6],
                format_func=lambda x: {1:'Single',2:'Married',3:'Widower',
                                        4:'Divorced',5:'Facto Union',6:'Legally Separated'}[x])
            gender = st.selectbox("Jenis Kelamin", options=[0,1],
                format_func=lambda x: {0:'Perempuan',1:'Laki-laki'}[x])

        with col2:
            age_at_enrollment = st.number_input("Usia saat Mendaftar", min_value=17, max_value=70, value=20)
            nationality = st.selectbox("Kewarganegaraan", options=[1,2,6,11,13,14,17,21,22,24,25,26,32,41,62,100,101,103,105,108,109],
                format_func=lambda x: { 1: 'Portugis',
                                        2: 'Jerman',
                                        6: 'Spanyol',
                                        11: 'Italia',
                                        13: 'Belanda',
                                        14: 'Inggris',
                                        17: 'Lithuania',
                                        21: 'Angola',
                                        22: 'Tanjung Verde',
                                        24: 'Guinea',
                                        25: 'Mozambik',
                                        26: 'Sao Tome',
                                        32: 'Turki',
                                        41: 'Brasil',
                                        62: 'Rumania',
                                        100: 'Moldova',
                                        101: 'Meksiko',
                                        103: 'Ukraina',
                                        105: 'Rusia',
                                        108: 'Kuba',
                                        109: 'Kolombia'
                                    }.get(x, str(x)))

        with col3:
            international = st.selectbox("Mahasiswa Internasional", options=[0,1],
                format_func=lambda x: {0:'Tidak',1:'Ya'}[x])
            displaced = st.selectbox("Mahasiswa Displaced", options=[0,1],
                format_func=lambda x: {0:'Tidak',1:'Ya'}[x])

        # ── Informasi Akademik ─────────────────────────────────────────────────
        st.markdown('<p class="section-title"> Latar Belakang Akademik</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            application_mode = st.selectbox("Mode Aplikasi", options=[1,17,18,39,42,43],
                format_func=lambda x: {1:'1st Phase General',17:'2nd Phase General',
                                        18:'3rd Phase General',39:'Over 23 Years',
                                        42:'Transfer',43:'Change of Course'}.get(x, str(x)))
            application_order = st.slider("Urutan Pilihan (0=Pertama)", 0, 9, 1)

        with col2:
            course = st.selectbox("Program Studi",
                options=[9147,9500,9119,9773,9070,9085,9238,9003,9254,9853,9991,9670,9556,171,33,9130,8014],
                format_func=lambda x: {9147:'Management',9500:'Nursing',
                                        9119:'Informatics Engineering',9773:'Journalism',
                                        9070:'Communication Design',9085:'Veterinary Nursing',
                                        9238:'Social Service',9003:'Agronomy',
                                        9254:'Tourism',9853:'Basic Education',
                                        9991:'Management (Evening)',9670:'Advertising & Marketing',
                                        9556:'Oral Hygiene',171:'Animation & Multimedia',
                                        33:'Biofuel Technologies',9130:'Equinculture',
                                        8014:'Social Service (Evening)'}.get(x, str(x)))
            daytime_evening = st.selectbox("Waktu Kuliah", options=[1,0],
                format_func=lambda x: {1:'Siang (Daytime)',0:'Malam (Evening)'}[x])

        with col3:
            prev_qualification = st.selectbox("Kualifikasi Sebelumnya", options=[1,2,3,4],
                format_func=lambda x: {1:'SMA',2:"Bachelor's",3:'Degree',4:"Master's"}[x])
            prev_qual_grade = st.slider("Nilai Kualifikasi Sebelumnya (0-200)", 0, 200, 130)
            admission_grade = st.slider("Nilai Penerimaan (0-200)", 0, 200, 130)

        # ── Informasi Orang Tua ────────────────────────────────────────────────
        st.markdown('<p class="section-title">Latar Belakang Orang Tua</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            mothers_qualification = st.selectbox("Pendidikan Ibu",
                options=[1,2,3,4,19,34,35,36,37,38,39,40,41,42,43,44],
                format_func=lambda x: {1:'SMA',2:'S1',3:'Sarjana',4:'S2',19:'SD',
                                        34:'Tidak Sekolah',35:'SD Kelas 4',36:'SMP',
                                        37:'SMP (Siklus 2)',38:'Teknologi',39:'Spesialisasi Teknologi',
                                        40:'S1 (Siklus 1)',41:'Spesialisasi',42:'Profesi',
                                        43:'S2 (Siklus 2)',44:'Doktor'}.get(x, str(x)), index=0)
            mothers_occupation = st.selectbox("Pekerjaan Ibu",
                options=[0,1,2,3,4,5,6,7,8,9,10],
                format_func=lambda x: {0:'Siswa',1:'Perwakilan Legislatif',2:'Spesialis Intelektual',
                                        3:'Teknisi',4:'Staf Administrasi',5:'Jasa & Keamanan',
                                        6:'Pertanian',7:'Industri & Konstruksi',8:'Operator Mesin',
                                        9:'Pekerja Tidak Terampil',10:'Profesi Angkatan Bersenjata'}.get(x, str(x)),
                index=0)

        with col2:
            fathers_qualification = st.selectbox("Pendidikan Ayah",
                options=[1,2,3,4,19,34,35,36,37,38,39,40,41,42,43,44],
                format_func=lambda x: {1:'SMA',2:'S1',3:'Sarjana',4:'S2',19:'SD',
                                        34:'Tidak Sekolah',35:'SD Kelas 4',36:'SMP',
                                        37:'SMP (Siklus 2)',38:'Teknologi',39:'Spesialisasi Teknologi',
                                        40:'S1 (Siklus 1)',41:'Spesialisasi',42:'Profesi',
                                        43:'S2 (Siklus 2)',44:'Doktor'}.get(x, str(x)), index=0)
            fathers_occupation = st.selectbox("Pekerjaan Ayah",
                options=[0,1,2,3,4,5,6,7,8,9,10],
                format_func=lambda x: {0:'Siswa',1:'Perwakilan Legislatif',2:'Spesialis Intelektual',
                                        3:'Teknisi',4:'Staf Administrasi',5:'Jasa & Keamanan',
                                        6:'Pertanian',7:'Industri & Konstruksi',8:'Operator Mesin',
                                        9:'Pekerja Tidak Terampil',10:'Profesi Angkatan Bersenjata'}.get(x, str(x)),
                index=0)

        # ── Kondisi Finansial ──────────────────────────────────────────────────
        st.markdown('<p class="section-title"> Kondisi Finansial</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tuition_fees = st.selectbox("Biaya Kuliah Lunas?", options=[1,0],
                format_func=lambda x: {1:'Ya',0:'Belum'}[x])
        with col2:
            debtor = st.selectbox("Memiliki Hutang?", options=[0,1],
                format_func=lambda x: {0:'Tidak',1:'Ya'}[x])
        with col3:
            scholarship = st.selectbox("Penerima Beasiswa?", options=[0,1],
                format_func=lambda x: {0:'Tidak',1:'Ya'}[x])
        with col4:
            special_needs = st.selectbox("Kebutuhan Khusus Pendidikan?", options=[0,1],
                format_func=lambda x: {0:'Tidak',1:'Ya'}[x])

        # ── Performa Akademik Semester 1 ───────────────────────────────────────
        st.markdown('<p class="section-title"> Performa Semester 1</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            cu1_credited = st.number_input("Unit Dikreditkan", min_value=0, max_value=20, value=0, key='cu1c')
            cu1_enrolled = st.number_input("Unit Diambil",     min_value=0, max_value=26, value=6, key='cu1e')
        with col2:
            cu1_evals    = st.number_input("Jumlah Evaluasi",  min_value=0, max_value=21, value=6, key='cu1ev')
            cu1_approved = st.number_input("Unit Disetujui",   min_value=0, max_value=26, value=5, key='cu1a')
        with col3:
            cu1_grade    = st.slider("Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, step=0.1, key='cu1g')
            cu1_no_eval  = st.number_input("Unit Tanpa Evaluasi", min_value=0, max_value=12, value=0, key='cu1ne')

        # ── Performa Akademik Semester 2 ───────────────────────────────────────
        st.markdown('<p class="section-title"> Performa Semester 2</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            cu2_credited = st.number_input("Unit Dikreditkan", min_value=0, max_value=20, value=0, key='cu2c')
            cu2_enrolled = st.number_input("Unit Diambil",     min_value=0, max_value=23, value=6, key='cu2e')
        with col2:
            cu2_evals    = st.number_input("Jumlah Evaluasi",  min_value=0, max_value=19, value=6, key='cu2ev')
            cu2_approved = st.number_input("Unit Disetujui",   min_value=0, max_value=20, value=5, key='cu2a')
        with col3:
            cu2_grade    = st.slider("Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, step=0.1, key='cu2g')
            cu2_no_eval  = st.number_input("Unit Tanpa Evaluasi", min_value=0, max_value=12, value=0, key='cu2ne')

        # ── Faktor Makroekonomi ────────────────────────────────────────────────
        st.markdown('<p class="section-title">Faktor Makroekonomi</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            unemployment_rate = st.number_input("Tingkat Pengangguran (%)",
                min_value=0.0, max_value=25.0, value=10.8, step=0.1)
        with col2:
            inflation_rate = st.number_input("Tingkat Inflasi (%)",
                min_value=-5.0, max_value=10.0, value=1.4, step=0.1)
        with col3:
            gdp = st.number_input("GDP", min_value=-10.0, max_value=10.0, value=1.74, step=0.01)

        # ── Submit ─────────────────────────────────────────────────────────────
        submitted = st.form_submit_button(" Prediksi Status Mahasiswa",
                                          use_container_width=True, type="primary")

    # ── Hasil Prediksi ─────────────────────────────────────────────────────────
    if submitted:
        # Buat DataFrame dari 36 kolom raw
        raw_input = pd.DataFrame([{
            'Marital_status':                              marital_status,
            'Application_mode':                            application_mode,
            'Application_order':                           application_order,
            'Course':                                      course,
            'Daytime_evening_attendance':                  daytime_evening,
            'Previous_qualification':                      prev_qualification,
            'Previous_qualification_grade':                prev_qual_grade,
            'Nacionality':                                 nationality,
            'Mothers_qualification':                       mothers_qualification,
            'Fathers_qualification':                       fathers_qualification,
            'Mothers_occupation':                          mothers_occupation,
            'Fathers_occupation':                          fathers_occupation,
            'Admission_grade':                             admission_grade,
            'Displaced':                                   displaced,
            'Educational_special_needs':                   special_needs,
            'Debtor':                                      debtor,
            'Tuition_fees_up_to_date':                     tuition_fees,
            'Gender':                                      gender,
            'Scholarship_holder':                          scholarship,
            'Age_at_enrollment':                           age_at_enrollment,
            'International':                               international,
            'Curricular_units_1st_sem_credited':           cu1_credited,
            'Curricular_units_1st_sem_enrolled':           cu1_enrolled,
            'Curricular_units_1st_sem_evaluations':        cu1_evals,
            'Curricular_units_1st_sem_approved':           cu1_approved,
            'Curricular_units_1st_sem_grade':              cu1_grade,
            'Curricular_units_1st_sem_without_evaluations':cu1_no_eval,
            'Curricular_units_2nd_sem_credited':           cu2_credited,
            'Curricular_units_2nd_sem_enrolled':           cu2_enrolled,
            'Curricular_units_2nd_sem_evaluations':        cu2_evals,
            'Curricular_units_2nd_sem_approved':           cu2_approved,
            'Curricular_units_2nd_sem_grade':              cu2_grade,
            'Curricular_units_2nd_sem_without_evaluations':cu2_no_eval,
            'Unemployment_rate':                           unemployment_rate,
            'Inflation_rate':                              inflation_rate,
            'GDP':                                         gdp,
        }])

        # Terapkan outlier clipping + feature engineering → 42 kolom
        input_engineered = apply_feature_engineering(raw_input)

        # Ambil urutan kolom sesuai feature_names.pkl (42 kolom)
        input_final = input_engineered[feature_names]

        # Prediksi via Pipeline (preprocessing sudah di dalam)
        pred_encoded = model.predict(input_final)[0]
        pred_proba   = model.predict_proba(input_final)[0]
        pred_label   = le.inverse_transform([pred_encoded])[0]

        st.divider()
        st.markdown("### Hasil Prediksi")

        result_class = {
            'Dropout':  'result-dropout',
            'Enrolled': 'result-enrolled',
            'Graduate': 'result-graduate'
        }.get(pred_label, 'result-enrolled')

        icon = {'Dropout':'🔴','Enrolled':'🟡','Graduate':'🟢'}.get(pred_label, '🔵')

        st.markdown(f"""
        <div class="{result_class}">
            <h2>{icon} Status Prediksi: {pred_label}</h2>
            <p style="font-size:1rem; color:#555; margin-top:0.5rem;">
                Berdasarkan data yang dimasukkan, mahasiswa ini diprediksi akan <strong>{pred_label}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("####  Probabilitas per Kelas")
        prob_df = pd.DataFrame({
            'Status':       le.classes_,
            'Probabilitas': pred_proba
        }).sort_values('Probabilitas', ascending=False)

        cols = st.columns(len(le.classes_))
        for i, (_, row) in enumerate(prob_df.iterrows()):
            with cols[i]:
                st.metric(label=row['Status'], value=f"{row['Probabilitas']*100:.1f}%")

        for _, row in prob_df.iterrows():
            st.progress(float(row['Probabilitas']), text=f"{row['Status']}: {row['Probabilitas']*100:.1f}%")

        # Rekomendasi
        st.markdown("####  Rekomendasi Tindakan")
        if pred_label == 'Dropout':
            st.error("""
            **Mahasiswa ini berisiko tinggi untuk DROPOUT.** Tindakan yang disarankan:
            - Segera jadwalkan sesi konseling akademik
            - Evaluasi kondisi finansial dan tawarkan skema keringanan biaya
            - Berikan pendampingan intensif dari dosen pembimbing
            - Hubungkan dengan program beasiswa yang tersedia
            """)
        elif pred_label == 'Enrolled':
            st.warning("""
             **Mahasiswa ini masih aktif namun perlu pemantauan.** Tindakan yang disarankan:
            - Lakukan monitoring berkala setiap semester
            - Pastikan mahasiswa mendapat dukungan akademik yang cukup
            - Dorong keterlibatan dalam kegiatan kampus
            """)
        else:
            st.success("""
             **Mahasiswa ini diprediksi akan LULUS.** Tindakan yang disarankan:
            - Pertahankan performa akademik yang baik
            - Tawarkan program pengembangan karir dan magang
            - Dorong untuk terlibat dalam penelitian atau proyek unggulan
            """)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – Prediksi Batch
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Prediksi Batch dari File CSV")
    st.caption("Upload file CSV yang berisi data beberapa mahasiswa sekaligus.")

    st.info("""
    **Format CSV:** Pastikan file CSV memiliki **36 kolom raw** berikut (tanpa kolom Status):

    `Marital_status, Application_mode, Application_order, Course,
    Daytime_evening_attendance, Previous_qualification, Previous_qualification_grade,
    Nacionality, Mothers_qualification, Fathers_qualification, Mothers_occupation,
    Fathers_occupation, Admission_grade, Displaced, Educational_special_needs,
    Debtor, Tuition_fees_up_to_date, Gender, Scholarship_holder, Age_at_enrollment,
    International, Curricular_units_1st_sem_credited, Curricular_units_1st_sem_enrolled,
    Curricular_units_1st_sem_evaluations, Curricular_units_1st_sem_approved,
    Curricular_units_1st_sem_grade, Curricular_units_1st_sem_without_evaluations,
    Curricular_units_2nd_sem_credited, Curricular_units_2nd_sem_enrolled,
    Curricular_units_2nd_sem_evaluations, Curricular_units_2nd_sem_approved,
    Curricular_units_2nd_sem_grade, Curricular_units_2nd_sem_without_evaluations,
    Unemployment_rate, Inflation_rate, GDP`

    Feature engineering (approval_rate, financial_risk, dll) akan dihitung **otomatis** oleh aplikasi.
    """)

    uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file, sep=';')
            if 'Status' in df_batch.columns:
                df_batch = df_batch.drop(columns=['Status'])
            if 'Status_encoded' in df_batch.columns:
                df_batch = df_batch.drop(columns=['Status_encoded'])

            st.success(f" File berhasil dimuat: **{df_batch.shape[0]} baris**, **{df_batch.shape[1]} kolom**")

            # Cek kolom raw yang dibutuhkan
            missing_cols = [c for c in RAW_COLS if c not in df_batch.columns]
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan di CSV: {missing_cols}")
            else:
                # Feature engineering → 42 kolom
                df_engineered = apply_feature_engineering(df_batch[RAW_COLS])
                df_input      = df_engineered[feature_names]

                predictions  = model.predict(df_input)
                probabilities = model.predict_proba(df_input)

                df_result = df_batch.copy()
                df_result['Predicted_Status'] = le.inverse_transform(predictions)

                for i, cls in enumerate(le.classes_):
                    df_result[f'Prob_{cls}'] = probabilities[:, i].round(4)

                # Summary
                st.markdown("#### Ringkasan Prediksi")
                col1, col2, col3 = st.columns(3)
                counts = pd.Series(le.inverse_transform(predictions)).value_counts()

                with col1:
                    n = counts.get('Dropout', 0)
                    st.markdown(f"""<div class="metric-card">
                        <div class="value" style="color:#c62828;">{n}</div>
                        <div class="label">🔴 Dropout ({n/len(predictions)*100:.1f}%)</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    n = counts.get('Enrolled', 0)
                    st.markdown(f"""<div class="metric-card">
                        <div class="value" style="color:#f57f17;">{n}</div>
                        <div class="label">🟡 Enrolled ({n/len(predictions)*100:.1f}%)</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    n = counts.get('Graduate', 0)
                    st.markdown(f"""<div class="metric-card">
                        <div class="value" style="color:#1b5e20;">{n}</div>
                        <div class="label">🟢 Graduate ({n/len(predictions)*100:.1f}%)</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("#### Tabel Hasil Prediksi")
                st.dataframe(
                    df_result[['Predicted_Status'] + [f'Prob_{c}' for c in le.classes_]].head(50),
                    use_container_width=True,
                    column_config={
                        "Predicted_Status": st.column_config.TextColumn("Status Prediksi"),
                        "Prob_Dropout":  st.column_config.ProgressColumn("P(Dropout)",  min_value=0, max_value=1),
                        "Prob_Enrolled": st.column_config.ProgressColumn("P(Enrolled)", min_value=0, max_value=1),
                        "Prob_Graduate": st.column_config.ProgressColumn("P(Graduate)", min_value=0, max_value=1),
                    }
                )

                # Download hasil
                csv_out = df_result.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "⬇Download Hasil Prediksi (CSV)",
                    csv_out, "hasil_prediksi.csv", "text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error memproses file: {str(e)}")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem 0;">
    <strong>Jaya Jaya Institut</strong> — Student Dropout Early Warning System<br>
</div>
""", unsafe_allow_html=True)