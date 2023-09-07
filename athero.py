import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time 
import matplotlib.pyplot as plt
from io import BytesIO

# Fungsi untuk membuat grafik
def create_monitor_graph(heart_rate_data, spo2_data):
    # Buat gambar dan sumbu
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot data Heart Rate
    ax.plot(heart_rate_data, label='Heart Rate (BPM)', color='red')

    # Plot data SPO2
    ax.plot(spo2_data, label='SPO2 (%)', color='blue')

    # Konfigurasi sumbu dan label
    ax.set_xlabel('Waktu')
    ax.set_ylabel('Nilai')
    ax.set_title('Monitor Heart Rate dan SPO2')
    ax.legend()

    # Simpan grafik ke dalam objek BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Data dummy untuk Heart Rate dan SPO2 (gantilah dengan data yang sesuai)
heart_rate_data = np.random.randint(60, 100, 100)
spo2_data = np.random.randint(90, 100, 100)

# Judul aplikasi
st.markdown('<style>h1{font-family: "Poppins", sans-serif;}</style>', unsafe_allow_html=True)
st.title("Deteksi Dini Atherosclerosis")

# Garis putus-putus untuk memberikan jarak
st.markdown("---")

# Membagi layar menjadi dua kolom untuk Heart Rate dan SPO2
col1, col2 = st.columns(2)

# Menambahkan margin bawah untuk kolom pertama
with col1:
    st.markdown('<div style="background-color: #2E8B57; text-align: center; padding: 20px; border-radius: 10px; margin-bottom: 20px;">Heart Rate : 75 BPM</div>', unsafe_allow_html=True)

# Menambahkan margin bawah untuk kolom kedua
with col2:
    st.markdown('<div style="background-color: #2E8B57; text-align: center; padding: 20px; border-radius: 10px; margin-bottom: 20px;">SPO2 : 95 %</div>', unsafe_allow_html=True)

# Buat grafik
graph = create_monitor_graph(heart_rate_data, spo2_data)

# Tampilkan grafik
st.image(graph, use_column_width=True)

# Tampilkan ekspander "Jenis Nyeri Dada"
with st.expander("Tipe Nyeri Dada"):
    st.write("""
    **Tipe 1. Nyeri Dada Seperti Sakit di Dada Tengah (Typical Angina)**
    - Kadang-kadang, dada kita bisa terasa berat atau seperti terbakar di bagian tengahnya. Ini bisa terjadi ketika kita berolahraga keras atau merasa sangat khawatir. Biasanya, jika kita istirahat atau minum obat tertentu, sakitnya akan hilang.

    **Tipe 2. Sakit di Dada yang Tidak Biasa (Atypical Angina)**
    - Terkadang, dada kita bisa terasa sakit tapi tidak seberat yang pertama. Sakitnya bisa di mana saja di dada kita, dan mungkin tidak terjadi setelah kita bergerak banyak atau merasa stres. Obat mungkin tidak selalu membuatnya hilang.

    **Tipe 3. Sakit di Dada yang Bukan Karena Masalah Jantung (Non-Anginal Pain)**
    - Ada juga sakit di dada yang bukan karena jantung kita. Ini bisa disebabkan oleh berbagai hal, seperti otot yang lelah, masalah pada perut, atau bahkan ketika kita batuk atau bernapas dalam-dalam. Ini bukan tanda-tanda masalah dengan jantung kita.

    **Tipe 4. Tidak Ada Sakit di Dada (Asymptomatic)**
    - Tidak merasa sakit di dada sama sekali, namun ini dapat terjadi meskipun ada masalah dengan jantung kita. Ini bisa terjadi pada orang dewasa yang lebih tua atau orang dengan diabetes, dan ini berarti kita harus lebih berhati-hati untuk menjaga kesehatan jantung kita.
    """)

# Tampilkan ekspander "Exercise Angina"
with st.expander("Exercise Angina"):
    st.write("Exercise angina adalah sakit di dada yang terjadi ketika kita bermain terlalu keras atau berolahraga dan jantung kita tidak bisa mengikuti. Jantung kita membutuhkan darah dan oksigen untuk bekerja dengan baik, tetapi kadang-kadang ada halangan di dalam pembuluh darah yang membawa darah dan oksigen ke jantung. Hal ini bisa disebabkan oleh kotoran yang menempel di dinding pembuluh darah, seperti lemak atau gula. Ketika kita bermain terlalu keras, jantung kita membutuhkan lebih banyak darah dan oksigen, tetapi halangan ini membuatnya sulit untuk sampai ke jantung. Akibatnya, jantung kita merasa kesulitan dan mengirim sinyal sakit ke dada kita. Sakit ini biasanya akan hilang jika kita berhenti bermain atau mengurangi kecepatan kita.")

# Muat model pertama dengan kolom 'Temperature', 'Heart_rate', 'SPO2'
random_forest_model1 = joblib.load("rf_model2.pkl")

# Muat model kedua dengan kolom 'Sex', 'Chest pain type', 'Max HR', 'Exercise angina', 'Heart Disease'
random_forest_model2 = joblib.load("rf_model1.pkl")

# Input dari pengguna untuk model kedua
st.sidebar.header("Masukkan Data")
sex = st.sidebar.radio("Jenis Kelamin", ["Pria", "Wanita"])
sex = 1 if sex == "Pria" else 0

# Input dari pengguna untuk model pertama
temperature = st.sidebar.number_input("Suhu (Temperature)", min_value=None, max_value=None, value=30)
heart_rate = st.sidebar.number_input("Denyut Jantung (Heart Rate)", min_value=None, max_value=None, value=75)
spo2 = st.sidebar.number_input("Tingkat SPO2", min_value=None, max_value=None, value=95)
chest_pain_type = st.sidebar.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4])
max_hr = st.sidebar.number_input("Denyut Jantung Maksimum (Max HR)", min_value=None, max_value=None, value=150)
exercise_angina = st.sidebar.radio("Apakah Anda Mengalami Exercise Angina?", ["Ya", "Tidak"])
exercise_angina = 1 if exercise_angina == "Ya" else 0

# Tampilkan tabel DataFrame input pengguna
# Tampilkan tabel DataFrame input pengguna
input_data_df = pd.DataFrame({
    'Kolom': ['Jenis Kelamin', 'Suhu (Temperature)', 'Denyut Jantung (Heart Rate)', 'Tingkat SPO2', 'Tipe Nyeri Dada', 'Denyut Jantung Maksimum (Max HR)', 'Exercise Angina'],
    'Input': ["Pria" if sex == 1 else "Wanita", temperature, heart_rate, spo2, chest_pain_type, max_hr, "Ya" if exercise_angina == 1 else "Tidak"]
})

# Mengubah indeks DataFrame agar dimulai dari 1 daripada 0
input_data_df.index = input_data_df.index + 1

st.subheader("Data Input Pengguna:")
st.write(input_data_df)

# Prediksi dilakukan hanya jika tombol "Prediksi" ditekan
if st.sidebar.button("Prediksi"):
    # Fungsi untuk memprediksi dengan model pertama
    def predict_with_model1(temperature, heart_rate, spo2):
        input_data = [[temperature, heart_rate, spo2]]
        prediction = random_forest_model1.predict(input_data)
        return prediction[0]

    # Fungsi untuk memprediksi dengan model kedua
    def predict_with_model2(sex, chest_pain_type, max_hr, exercise_angina):
        input_data = [[sex, chest_pain_type, max_hr, exercise_angina]]
        prediction = random_forest_model2.predict(input_data)
        return prediction[0]

    # Melakukan prediksi dengan model pertama
    prediction1 = predict_with_model1(temperature, heart_rate, spo2)

    # Melakukan prediksi dengan model kedua
    prediction2 = predict_with_model2(sex, chest_pain_type, max_hr, exercise_angina)

    # Memberikan bobot pada masing-masing model
    weight_model1 = 0.3 # Contoh bobot untuk model pertama
    weight_model2 = 0.7  # Contoh bobot untuk model kedua

    # Menggabungkan hasil prediksi dari kedua model (menggunakan aturan mayoritas)
    final_prediction = (weight_model1 * prediction1 + weight_model2 * prediction2) / (weight_model1 + weight_model2)

    # Menampilkan hasil prediksi akhir
    st.subheader("Hasil Prediksi:")
    if final_prediction <= 0.5:
        st.write("Status: Normal")
        st.success("Hasil prediksi menunjukkan kondisi yang normal. Namun, jika Anda memiliki kekhawatiran atau perlu dukungan tambahan, kami sarankan untuk berkonsultasi dengan spesialis atau dukungan medis sesuai kebutuhan.")
    else:
        st.write("Status: Tidak Normal")
        st.warning("Hasil prediksi menunjukkan potensi indikasi Atherosclerosis. Namun, perlu diingat bahwa ini hanya alat prediksi dan tidak menggantikan konsultasi medis profesional. Disarankan untuk berkonsultasi dengan dokter untuk evaluasi lebih lanjut dan diagnosis yang akurat.")
