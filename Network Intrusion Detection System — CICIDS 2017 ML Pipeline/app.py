import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# CACHE MODEL + PREPROCESSORS
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, selector, feature_names

model, scaler, selector, feature_names = load_model()

# -------------------------------
# UI HEADER
# -------------------------------
st.title("🛡️ Network Intrusion Detection System (CICIDS 2017)")
st.write("Fast ML-based threat detection using Random Forest")

# -------------------------------
# OPTION 1: FILE UPLOAD
# -------------------------------
st.subheader("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

uploaded_file = st.file_uploader("Upload your CICIDS CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # -------------------------------
    # CLEANING
    # -------------------------------
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # ensure correct features only
    X = data[feature_names]

    # -------------------------------
    # PREPROCESSING PIPELINE
    # -------------------------------
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    y_prob = model.predict_proba(X_selected)[:, 1]

    # -------------------------------
    # RISK ENGINE
    # -------------------------------
    def risk_label(p):
        score = p * 100
        if score >= 70:
            return "BLOCK"
        elif score >= 30:
            return "MONITOR"
        else:
            return "ALLOW"

    actions = [risk_label(p) for p in y_prob]

    # -------------------------------
    # OUTPUT
    # -------------------------------
    data["Risk Score"] = y_prob * 100
    data["Action"] = actions

    st.write("### 🔍 Predictions")
    st.dataframe(data)

    st.success("Batch prediction completed successfully!")

# -------------------------------
# OPTION 2: SINGLE PREDICTION
# -------------------------------
st.write("---")
st.subheader("🎯 Single Sample Prediction")

input_data = []

with st.form("single_input_form"):

    for col in feature_names:
        val = st.number_input(col, value=0.0)
        input_data.append(val)

    submit = st.form_submit_button("Predict")

if submit:

    X = np.array(input_data).reshape(1, -1)

    # preprocessing
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    # prediction
    prob = model.predict_proba(X_selected)[0][1]
    score = prob * 100

    if score >= 70:
        action = "BLOCK"
    elif score >= 30:
        action = "MONITOR"
    else:
        action = "ALLOW"

    st.success(f"Risk Score: {score:.2f}")
    st.success(f"Action: {action}")