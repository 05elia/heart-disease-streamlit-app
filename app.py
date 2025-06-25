import streamlit as st
import pandas as pd
import joblib

# === Load model dan scaler ===
model_rf = joblib.load("random_forest_model.pkl")
model_dt = joblib.load("decision_tree_model.pkl")
model_knn = joblib.load("knn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# === Streamlit UI ===
st.title("Prediksi Risiko Penyakit Jantung")
st.write("Masukkan data medis untuk memprediksi apakah pasien berisiko terkena penyakit jantung.")

# Pilih model
model_choice = st.selectbox("Pilih Model Machine Learning", ["Random Forest", "Decision Tree", "k-NN"])

# Input data pasien
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", step=1.0)

# Tombol prediksi
if st.button("Prediksi"):
    # Urutkan dan bentuk DataFrame
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # === DEBUG (cek input user dan scaling) ===
    st.subheader("📋 Data yang Anda Masukkan:")
    st.write(input_df)

    input_scaled = scaler.transform(input_df)

    # DEBUG scaled version
    st.subheader("🔧 Data Setelah Scaling:")
    st.write(pd.DataFrame(input_scaled, columns=feature_names))

    # Pilih model sesuai pilihan
    model_map = {
        "Random Forest": model_rf,
        "Decision Tree": model_dt,
        "k-NN": model_knn
    }
    model = model_map[model_choice]

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    st.subheader("🧠 Output Prediksi (angka):")
    st.write(prediction)

    result = "Berisiko" if prediction == 1 else "Tidak Berisiko"
    st.success(f"💡 Hasil Prediksi dengan {model_choice}: Pasien **{result}** terkena penyakit jantung.")
