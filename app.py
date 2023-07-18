import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from jcopml.pipeline import num_pipe
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeli ne

# Load dataset
df = pd.read_csv('KondisiSuhuUdaraKet.csv', delimiter=',')

# Pisahkan atribut dan target
x = df.drop(columns='keterangan')
y = df['keterangan']

# Bagi dataset menjadi data latih dan data uji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Definisikan transformer untuk preprocessing numerik
transformer = ColumnTransformer([
    ('numeric', num_pipe(), ['Karbon Monoksida', 'Sensitivitas Gas CO', 'Konsentrasi NMHC', 'Konsentrasi Benzena',
                             'Sensitivitas Gas Non-Methane Hydrocarbon (NMHC)', 'Konsentrasi NOx', 'Sensitivitas Gas NOx',
                             'Konsentrasi NO2', 'Sensitivitas Gas NO2', 'Sensitivitas Gas O3', 'Suhu', 'Kelembaban', 'Absolute Humidity'])
])

# Buat pipeline dengan preprocessing numerik dan model Gaussian Naive Bayes
pipeline = Pipeline([
    ('preprocess', transformer),
    ('model', GaussianNB())
])

# Latih model
pipeline.fit(x_train, y_train)

# Buat tampilan aplikasi Streamlit
def main():
    st.title("Aplikasi Prediksi Kondisi Udara")
    st.write("Masukkan data untuk melakukan prediksi kondisi udara")

    # Buat input form dengan 4 kolom
    col1, col2, col3 = st.columns(3)
    
    with col1:
        karbon_monoksida = st.number_input("Karbon Monoksida")
        sensitivitas_gas_co = st.number_input("Sensitivitas Gas CO")
        konsentrasi_nmhc = st.number_input("Konsentrasi NMHC")
        konsentrasi_benzena = st.number_input("Konsentrasi Benzena")
        
    with col2:
        sensitivitas_gas_nmhc = st.number_input("Sensitivitas Gas Non-Methane Hydrocarbon (NMHC)")
        konsentrasi_nox = st.number_input("Konsentrasi NOx")
        sensitivitas_gas_nox = st.number_input("Sensitivitas Gas NOx")
        konsentrasi_no2 = st.number_input("Konsentrasi NO2")
        
    with col3:
        sensitivitas_gas_no2 = st.number_input("Sensitivitas Gas NO2")
        sensitivitas_gas_o3 = st.number_input("Sensitivitas Gas O3")
        suhu = st.number_input("Suhu")
        kelembaban = st.number_input("Kelembaban")
        
    absolute_humidity = st.number_input("Absolute Humidity")
   

    # Buat dataframe dari input pengguna
    input_data = pd.DataFrame({
        'Karbon Monoksida': [karbon_monoksida],
        'Sensitivitas Gas CO': [sensitivitas_gas_co],
        'Konsentrasi NMHC': [konsentrasi_nmhc],
        'Konsentrasi Benzena': [konsentrasi_benzena],
        'Sensitivitas Gas Non-Methane Hydrocarbon (NMHC)': [sensitivitas_gas_nmhc],
        'Konsentrasi NOx': [konsentrasi_nox],
        'Sensitivitas Gas NOx': [sensitivitas_gas_nox],
        'Konsentrasi NO2': [konsentrasi_no2],
        'Sensitivitas Gas NO2': [sensitivitas_gas_no2],
        'Sensitivitas Gas O3': [sensitivitas_gas_o3],
        'Suhu': [suhu],
        'Kelembaban': [kelembaban],
        'Absolute Humidity': [absolute_humidity]
    })
# Buat tombol prediksi
    if st.button("Prediksi"):
        # Prediksi dengan menggunakan model
        prediksi = pipeline.predict(input_data)

        # Tampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        st.write(prediksi)

if __name__ == '__main__':
    main()
