import streamlit as st
import pandas as pd
import joblib

# === Load model dan scaler ===
model_rf = joblib.load("random_forest_model.pkl")
model_dt = joblib.load("decision_tree_model.pkl")
model_knn = joblib.load("knn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# === Sidebar ===
st.sidebar.title("📘 Informasi Penelitian")
st.sidebar.markdown("### 💻 Judul Penelitian:")
st.sidebar.markdown("**Klasifikasi untuk Mendeteksi Penyakit Jantung**")
st.sidebar.markdown("### 👥 Kelompok 6:")
anggota = [
    "Elia Jose Alvaro Rahayaan",
    "Aminatul Maimunah Al-Amalah",
    "Syahril Gibran Wangsaguna",
    "Regiyan Dwi Anugerah",
    "Firham Syahid"
]
for nama in anggota:
    st.sidebar.markdown(f"- {nama}")

st.sidebar.markdown("---")
st.sidebar.info("Masukkan data di halaman utama untuk melihat hasil prediksi.")

# === Header Utama ===
st.markdown("<h1 style='color:#1f77b4;'>🫀 Prediksi Risiko Penyakit Jantung</h1>", unsafe_allow_html=True)
st.write("Gunakan data medis pasien untuk memprediksi risiko penyakit jantung menggunakan model Machine Learning.")

# === Pilih Model ===
model_choice = st.selectbox("🔍 Pilih Model Machine Learning", ["Random Forest", "Decision Tree", "k-NN"])

# === Input Data Pasien ===
st.markdown("### 📝 Masukkan Data Pasien:")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", step=1.0)

# === Prediksi ===
if st.button("🔮 Prediksi"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    st.markdown("#### 📋 Data yang Dimasukkan:")
    st.dataframe(input_df)

    input_scaled = scaler.transform(input_df)
    st.markdown("#### 🔧 Data Setelah Scaling:")
    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))

    # Prediksi
    model_map = {
        "Random Forest": model_rf,
        "Decision Tree": model_dt,
        "k-NN": model_knn
    }
    model = model_map[model_choice]
    prediction = model.predict(input_scaled)[0]
    result = "Berisiko" if prediction == 1 else "Tidak Berisiko"

    st.markdown("### 🧠 Hasil Prediksi:")
    st.success(f"Model **{model_choice}** memprediksi pasien: **{result}** terhadap penyakit jantung.")
