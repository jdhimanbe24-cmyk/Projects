import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
from keras.models import load_model
import os

# =========================
# CONFIG
# =========================
SEQ_LENGTH = 30
MAX_RUL = 125

st.set_page_config(page_title="RUL Prediction", layout="wide")

st.title("🔧 Remaining Useful Life (RUL) Prediction")
st.write("Upload engine sensor data to predict RUL")

# =========================
# DEBUG (VERY IMPORTANT)
# =========================
st.sidebar.write("📂 Current dir:", os.getcwd())
st.sidebar.write("📂 Files:", os.listdir())

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_and_tools(model_path, scaler_path, feature_path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path, "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

# =========================
# SEQUENCE FUNCTION
# =========================
def create_sequences(df, seq_length, feature_cols):
    X = []
    values = df[feature_cols].values

    for i in range(len(df) - seq_length + 1):
        X.append(values[i:i+seq_length])

    return np.array(X)

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("⚙️ Settings")

dataset_choice = st.sidebar.selectbox(
    "Select Dataset Model",
    ["FD001", "FD002", "FD003", "FD004"]
)

# ✅ FIX: Works whether files are in folder OR same directory
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "bilstm_" + dataset_choice + ".keras")
scaler_path = os.path.join(BASE_DIR, "scaler_" + dataset_choice + ".pkl")
feature_path = os.path.join(BASE_DIR, "features_" + dataset_choice + ".json")

# =========================
# LOAD MODEL
# =========================
try:
    model, scaler, feature_cols = load_model_and_tools(
        model_path, scaler_path, feature_path
    )
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    st.sidebar.error("❌ Model not found")
    st.sidebar.write("Error:", e)
    st.stop()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    # =========================
    # CHECK FEATURES
    # =========================
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if len(missing_cols) > 0:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()

    # =========================
    # PREPROCESS
    # =========================
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])

    # =========================
    # CREATE SEQUENCES
    # =========================
    if len(df) < SEQ_LENGTH:
        st.error(f"❌ Need at least {SEQ_LENGTH} rows")
        st.stop()

    X = create_sequences(df, SEQ_LENGTH, feature_cols)

    # =========================
    # PREDICT
    # =========================
    preds = model.predict(X).flatten()
    preds_real = preds * MAX_RUL

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📈 Predictions")

    result_df = pd.DataFrame({
        "Predicted_RUL": preds_real
    })

    st.dataframe(result_df)

    # Latest prediction (important)
    st.metric("🧠 Latest RUL Prediction", f"{preds_real[-1]:.2f} cycles")

    # =========================
    # VISUALIZATION
    # =========================
    st.line_chart(preds_real)

else:
    st.info("👆 Upload a CSV file to start prediction")
