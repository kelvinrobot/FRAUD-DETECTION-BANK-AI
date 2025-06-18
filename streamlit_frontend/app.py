import sys
import os
import io
import requests
import shap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import local explainability modules
from explainability.shap_dense import explain_dense, FEATURE_NAMES
from explainability.ae_explain import explain_autoencoder
from explainability.lstm_explain import explain_lstm

# Streamlit Config 
st.set_page_config(page_title=" AI Fraud Detection", layout="wide")
st.title("🤖 AI Fraud Detection System")
st.write("Upload a CSV or enter transaction data manually to detect fraud and view model explanations.")

# PDF buffers
buf_shap, buf_ae, buf_lstm = io.BytesIO(), io.BytesIO(), io.BytesIO()
pdf_buf = io.BytesIO()


# UTILITY: Display Scores 
def display_metrics(result):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud Detected", str(result["fraud_detected"]))
    col2.metric("Dense Score", round(result["dense_score"], 4))
    col3.metric("Autoencoder Error", round(result["autoencoder_error"], 4))
    col4.metric("LSTM Score", round(result["lstm_score"], 4))


# UTILITY: Generate Explanations 
def generate_explanations(row_data):
    st.subheader(" Explainability Dashboard")

    try:
        # SHAP (Dense)
        st.markdown("###  Dense Model (SHAP Waterfall)")
        shap_values = explain_dense(row_data)
        shap_exp = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=shap_values.data[0],
            feature_names=FEATURE_NAMES
        )
        fig_shap, _ = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        st.pyplot(fig_shap)
        fig_shap.savefig(buf_shap, format="png")

        # Autoencoder
        st.markdown("### 🛠 Autoencoder (Reconstruction Errors)")
        ae_error = explain_autoencoder(row_data)
        fig_ae, ax_ae = plt.subplots(figsize=(12, 5))
        ax_ae.bar(FEATURE_NAMES, ae_error)
        ax_ae.set_ylabel("Error")
        ax_ae.set_title("Feature-wise Reconstruction Error")
        ax_ae.set_xticklabels(FEATURE_NAMES, rotation=90)
        st.pyplot(fig_ae)
        fig_ae.savefig(buf_ae, format="png")

        # LSTM
        st.markdown("###  LSTM Model (Simulated Feature Impact)")
        lstm_delta = explain_lstm(row_data)
        fig_lstm, ax_lstm = plt.subplots(figsize=(12, 5))
        ax_lstm.plot(FEATURE_NAMES, lstm_delta, marker='o')
        ax_lstm.set_ylabel("Delta")
        ax_lstm.set_title("Simulated Feature Change Impact")
        ax_lstm.set_xticklabels(FEATURE_NAMES, rotation=90)
        st.pyplot(fig_lstm)
        fig_lstm.savefig(buf_lstm, format="png")

    except Exception as e:
        st.error(f" Explainability failed: {e}")


# UTILITY: PDF Export 
def download_pdf():
    try:
        with PdfPages(pdf_buf) as pdf:
            for buf in [buf_shap, buf_ae, buf_lstm]:
                buf.seek(0)
                img = plt.imread(buf)
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)

        st.download_button(
            label=" Download Explanations as PDF",
            data=pdf_buf.getvalue(),
            file_name="explanations.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f" PDF generation failed: {e}")


# CSV UPLOAD MODE 
st.subheader(" Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV with 29 transaction features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Data Preview")
    st.dataframe(df.head())

    if list(df.columns) != FEATURE_NAMES:
        st.error(" Column names in CSV do not match required 29 feature names.")
        st.stop()

    selected_row = st.selectbox(" Select a row to explain", range(len(df)))
    row_data = df.iloc[selected_row].tolist()

    if st.button(" Predict & Explain Selected Row"):
        with st.spinner("Sending data to backend..."):
            response = requests.post("http://localhost:8000/predict", json={"features": row_data})

        if response.status_code == 200:
            result = response.json()
            st.success(" Prediction Complete!")
            display_metrics(result)
            generate_explanations(row_data)
            download_pdf()
        else:
            st.error(f" API Error: {response.text}")
    st.stop()


# MANUAL INPUT MODE 
st.subheader(" Enter Features Manually")
with st.form("manual_input_form"):
    manual_features = []
    cols = st.columns(3)

    for i in range(29):
        col = cols[i % 3]
        val = col.number_input(FEATURE_NAMES[i], value=0.0, step=0.01)
        manual_features.append(val)

    submitted = st.form_submit_button(" Predict Manually")

if submitted:
    with st.spinner("Sending manual input to backend..."):
        response = requests.post("http://localhost:8000/predict", json={"features": manual_features})

    if response.status_code == 200:
        result = response.json()
        st.success(" Prediction Complete!")
        display_metrics(result)
        generate_explanations(manual_features)
        download_pdf()
    else:
        st.error(f" API Error: {response.text}")
