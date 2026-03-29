# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. Namun di balik pencapaian tersebut, institusi ini menghadapi tantangan serius: tingginya angka mahasiswa yang tidak menyelesaikan pendidikannya alias **dropout**.

Tingginya angka dropout berdampak langsung pada reputasi institusi, akreditasi program studi, serta efisiensi penggunaan sumber daya pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin mahasiswa yang berpotensi dropout sehingga dapat diberikan bimbingan dan intervensi khusus sebelum terlambat.

---

### Permasalahan Bisnis

1. **Tingginya angka dropout mahasiswa** — sebesar **32%** dari total 4.424 mahasiswa tercatat dropout, angka yang signifikan bagi keberlangsungan institusi.
2. **Tidak adanya sistem deteksi dini** — institusi belum memiliki mekanisme untuk mengidentifikasi mahasiswa berisiko dropout secara proaktif sebelum mereka benar-benar keluar.
3. **Faktor risiko yang belum terpetakan** — belum diketahui secara jelas faktor akademik, finansial, maupun demografis mana yang paling berkontribusi terhadap dropout.
4. **Keterbatasan monitoring berbasis data** — manajemen membutuhkan dashboard yang mudah dipahami untuk memantau performa dan distribusi status mahasiswa secara real-time.

---

### Cakupan Proyek

- Eksplorasi dan analisis data mahasiswa (EDA) untuk memahami distribusi status dan faktor-faktor yang berkaitan dengan dropout.
- Feature engineering untuk meningkatkan kualitas prediksi model (6 fitur baru: `approval_rate_1st`, `approval_rate_2nd`, `total_approved`, `avg_grade`, `financial_risk`, `enrolled_total`).
- Pembangunan dan evaluasi model machine learning **klasifikasi biner** (Dropout vs Graduate) menggunakan lima algoritma: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan XGBoost. Data mahasiswa berstatus Enrolled dikecualikan dari training karena belum memiliki label akhir.
- Pemilihan model terbaik berdasarkan metrik F1-Weighted yang relevan untuk data tidak seimbang.
- Pembuatan business dashboard interaktif menggunakan Metabase untuk monitoring status mahasiswa.
- Deployment prototype sistem prediksi berbasis web menggunakan Streamlit.

---

### Persiapan

**Sumber data:**
Dataset Students' Performance dari Jaya Jaya Institut — [https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Setup environment:**

```bash
# 1. Clone repositori atau ekstrak folder submission
# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install seluruh dependensi
pip install -r requirements.txt

# 4. Jalankan notebook untuk melatih ulang model (opsional)
jupyter notebook notebook.ipynb

# 5. Jalankan aplikasi Streamlit secara lokal
streamlit run app.py
```

---

## Business Dashboard

Dashboard dibuat menggunakan **Metabase** dan menampilkan visualisasi komprehensif untuk memonitor status mahasiswa serta faktor-faktor yang berpengaruh terhadap dropout.

**Konten dashboard mencakup:**

| Visualisasi | Deskripsi |
|---|---|
| Distribusi Status Mahasiswa | Donut chart proporsi Dropout (32%), Graduate (50%), Enrolled (18%) |
| Distribusi Usia Saat Pendaftaran | Bar chart usia per kelompok, dipecah per status |
| Status Pembayaran Tuition vs Dropout | Perbandingan mahasiswa tepat bayar vs menunggak terhadap status |
| Pemegang Beasiswa vs Dropout Rate | Efek beasiswa sebagai faktor protektif terhadap dropout |
| Tingkat Dropout per Jurusan | Bar chart jumlah dropout dan total mahasiswa per program studi |

**Akses Metabase (lokal dengan Docker):**
- **Email:** `hamzahbaik9@gmail.com`
- **Password:** `root123`


**Cara menjalankan Metabase dengan Docker:**
Untuk menjalankan dashboard secara lokal, ikuti langkah berikut:
 
```bash
# 1. Jalankan container Metabase
docker run -d -p 3000:3000 --name metabase metabase/metabase
 
# 2. Tunggu 2-3 menit, lalu copy file database
docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db
 
# 3. Restart container
docker restart metabase
 
# 4. Buka browser di http://localhost:3000
```
---

## Menjalankan Sistem Machine Learning

Sistem machine learning telah di-deploy sebagai aplikasi web interaktif menggunakan **Streamlit**. Aplikasi ini mendukung dua mode prediksi:

1. **Prediksi Individual** — Input data satu mahasiswa melalui form interaktif, sistem akan mengeluarkan prediksi status beserta probabilitas per kelas dan rekomendasi tindakan.
2. **Prediksi Batch** — Upload file CSV berisi data banyak mahasiswa sekaligus, hasil prediksi dapat diunduh kembali dalam format CSV.

**Akses prototype online:**
🔗 [https://student-dropout-early-jaya.streamlit.app/](https://student-dropout-early-jaya.streamlit.app/)

**Menjalankan secara lokal:**
```bash
# Pastikan folder model/ berisi file berikut:
# - best_model.pkl
# - label_encoder.pkl
# - feature_names.pkl
# - model_info.pkl

streamlit run app.py
# Akses di browser: http://localhost:8501
```

**Format input CSV untuk prediksi batch:**
File CSV harus menggunakan separator `;` dan memiliki 36 kolom berikut (tanpa kolom `Status`):
```
Marital_status, Application_mode, Application_order, Course,
Daytime_evening_attendance, Previous_qualification, Previous_qualification_grade,
Nacionality, Mothers_qualification, Fathers_qualification,
Mothers_occupation, Fathers_occupation, Admission_grade, Displaced,
Educational_special_needs, Debtor, Tuition_fees_up_to_date, Gender,
Scholarship_holder, Age_at_enrollment, International,
Curricular_units_1st_sem_credited, Curricular_units_1st_sem_enrolled,
Curricular_units_1st_sem_evaluations, Curricular_units_1st_sem_approved,
Curricular_units_1st_sem_grade, Curricular_units_1st_sem_without_evaluations,
Curricular_units_2nd_sem_credited, Curricular_units_2nd_sem_enrolled,
Curricular_units_2nd_sem_evaluations, Curricular_units_2nd_sem_approved,
Curricular_units_2nd_sem_grade, Curricular_units_2nd_sem_without_evaluations,
Unemployment_rate, Inflation_rate, GDP
```
Feature engineering akan dihitung **otomatis** oleh aplikasi.

---

## Conclusion

### Kesimpulan 1 — Faktor yang Berkaitan dengan Dropout (Berdasarkan EDA)

Berdasarkan analisis data eksploratif terhadap 4.424 mahasiswa Jaya Jaya Institut, ditemukan pola-pola berikut yang berkaitan erat dengan kejadian dropout:

**1. Faktor Finansial**

Mahasiswa yang menunggak biaya kuliah (`Tuition_fees_up_to_date = 0`) mendominasi kelompok Dropout — **86.6%** dari 528 mahasiswa yang menunggak berstatus Dropout (457 orang), jauh lebih tinggi dibanding mahasiswa tepat bayar yang hanya **~25%** Dropout (964 dari 3.896 mahasiswa). Selain itu, mahasiswa non-penerima beasiswa memiliki tingkat dropout **38.7%** (1.287 dari 3.325), sementara penerima beasiswa hanya **12.2%** (134 dari 1.099) — membuktikan beasiswa sebagai faktor protektif yang signifikan. Mahasiswa berstatus debtor juga menunjukkan kecenderungan dropout yang lebih tinggi.

**2. Faktor Akademik**

Mahasiswa Dropout secara konsisten memiliki jumlah mata kuliah lulus dan nilai rata-rata yang jauh lebih rendah di kedua semester dibanding kelompok Graduate. Pola ini sudah terlihat sejak semester 1 dan semakin tajam di semester 2, mengindikasikan bahwa performa akademik semester awal adalah sinyal peringatan dini yang krusial.

**3. Faktor Usia**

Rata-rata usia mahasiswa Dropout (26.1 tahun, median 23) jauh lebih tinggi dibanding Graduate (21.8 tahun, median 19). Mahasiswa yang lebih tua saat mendaftar cenderung memiliki tanggung jawab lain di luar kampus (pekerjaan, keluarga) yang meningkatkan risiko dropout.

---

### Kesimpulan 2 — Performa Model Machine Learning

Proyek ini membangun model **klasifikasi biner** (Dropout vs Graduate) menggunakan data 3.630 mahasiswa yang sudah memiliki label final. Data mahasiswa berstatus Enrolled (794 orang) dikecualikan dari training karena statusnya belum final dan dapat dimanfaatkan pada tahap inferensi. Dataset dibagi menjadi training set 2.904 sampel (80%) dan test set 726 sampel (20%).

**Performa lima model pada test set:**

| Model | Accuracy | F1-Weighted | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 91.1% | 0.9101 | 0.9502 |
| Decision Tree | 87.5% | 0.8745 | 0.8612 |
| **Random Forest** ✅ | **91.6%** | **0.9148** | **0.9570** |
| Gradient Boosting | 90.8% | 0.9069 | 0.9521 |
| XGBoost | 91.2% | 0.9111 | 0.9538 |

Model terbaik adalah **Random Forest** dengan F1-Weighted tertinggi (0.9148) dan ROC-AUC tertinggi (0.9570), unggul di semua metrik dibanding model lainnya. F1-Weighted diprioritaskan sebagai metrik seleksi karena lebih relevan untuk dataset dengan distribusi kelas tidak seimbang.

**Performa per kelas (Random Forest, test set 726 sampel):**
- **Dropout** — precision 0.95, recall 0.83, F1-score 0.89: model sangat presisi dalam memprediksi Dropout; recall 0.83 berarti 17% kasus Dropout aktual tidak terdeteksi
- **Graduate** — precision 0.90, recall 0.97, F1-score 0.93: hampir semua mahasiswa Graduate berhasil diidentifikasi dengan benar

**Fitur yang paling berkontribusi terhadap performa model** (feature importance Random Forest):

> Catatan: feature importance menjelaskan kontribusi setiap fitur terhadap kemampuan model membuat prediksi — bukan secara langsung menyimpulkan penyebab dropout. Faktor yang berkaitan dengan dropout diidentifikasi dari EDA (Kesimpulan 1).

| Rank | Fitur | Importance |
|---|---|---|
| 1 | `approval_rate_2nd` | 0.1624 |
| 2 | `Curricular_units_2nd_sem_approved` | 0.1050 |
| 3 | `total_approved` | 0.0983 |
| 4 | `approval_rate_1st` | 0.0865 |
| 5 | `Curricular_units_2nd_sem_grade` | 0.0580 |
| 6 | `avg_grade` | 0.0553 |
| 7 | `Curricular_units_1st_sem_approved` | 0.0421 |
| 8 | `Curricular_units_1st_sem_grade` | 0.0383 |
| 9 | `Tuition_fees_up_to_date` | 0.0275 |
| 10 | `financial_risk` | 0.0251 |

---

### Rekomendasi Action Items

- **Implementasi Early Warning System** — Bangun sistem peringatan otomatis yang memantau `approval_rate_2nd` dan `Curricular_units_2nd_sem_approved` di akhir setiap semester 2; mahasiswa dengan rasio lulus di bawah ambang batas tertentu segera mendapatkan notifikasi untuk intervensi.
- **Deteksi Dini Sejak Semester 1** — Pantau `approval_rate_1st` dan `Curricular_units_1st_sem_approved` sejak semester pertama sebagai sinyal awal risiko; penurunan performa sejak awal perlu ditindaklanjuti segera.
- **Program Keringanan dan Cicilan Tuition** — Buat skema pembayaran biaya kuliah yang fleksibel khusus bagi mahasiswa yang menunggak; data EDA menunjukkan 86.6% mahasiswa menunggak berakhir Dropout.
- **Perluasan Program Beasiswa** — Tingkatkan jumlah penerima beasiswa, terutama bagi mahasiswa dengan skor `financial_risk` tinggi; tingkat dropout penerima beasiswa (12.2%) jauh lebih rendah dari non-penerima (38.7%).
- **Program Khusus Mahasiswa Dewasa (>25 Tahun)** — Rata-rata usia mahasiswa Dropout (26.1 tahun) jauh lebih tinggi dibanding Graduate (21.8 tahun); sediakan program kelas malam, pembelajaran hybrid, atau skema studi paruh waktu.
- **Audit Program Studi dengan Dropout Tinggi** — Evaluasi kurikulum dan beban studi pada jurusan dengan tingkat dropout tertinggi untuk mengidentifikasi hambatan struktural.
