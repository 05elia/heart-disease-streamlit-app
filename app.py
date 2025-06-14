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
st.write("Gunakan data medis pasien untuk memprediksi risiko penyakit jantung.")

# Pilih model
model_choice = st.selectbox("Pilih Model Machine Learning", ["Random Forest", "Decision Tree", "k-NN"])

# Input fitur
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", step=1.0)

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([user_input], columns=feature_names)  # ✅ fix urutan
    input_scaled = scaler.transform(input_df)
    
    model = {
        "Random Forest": model_rf,
        "Decision Tree": model_dt,
        "k-NN": model_knn
    }[model_choice]

    prediction = model.predict(input_scaled)[0]
    result = "Berisiko" if prediction == 1 else "Tidak Berisiko"

    st.success(f"Hasil Prediksi ({model_choice}): Pasien **{result}** terkena penyakit jantung.")
