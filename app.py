import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Credit Risk Scoring", layout="centered")

st.title("💳 Credit Risk Scoring System")
st.write("Predict the probability of loan default using Machine Learning.")

st.markdown("---")

# Inputs
loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
annual_inc = st.number_input("Annual Income", min_value=0.0, value=50000.0)
fico = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0)

# Predict button
if st.button("Predict Risk"):

    # Build minimal input
    input_data = pd.DataFrame({
        "loan_amnt": [loan_amnt],
        "annual_inc": [annual_inc],
        "fico_range_low": [fico]
    })

    # Match model feature count
    while input_data.shape[1] < model.n_features_in_:
        input_data[f"extra_{input_data.shape[1]}"] = 0

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prob = model.predict_proba(input_scaled)[:,1][0]

    # Risk Band
    if prob < 0.30:
        band = "🟢 Low Risk"
    elif prob < 0.60:
        band = "🟠 Medium Risk"
    else:
        band = "🔴 High Risk"

    # Output
    st.success("Prediction Complete")

    st.metric("Default Probability", f"{prob:.2%}")
    st.write("Risk Category:", band)

st.markdown("---")
st.caption("Built with Python, XGBoost, Streamlit")
