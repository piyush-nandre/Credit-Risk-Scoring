import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit Header / Toolbar / Menu */
header {
    visibility: hidden;
    height: 0rem;
}

[data-testid="stHeader"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Remove top empty space */
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.5rem;
    max-width: 900px;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    color: #111827;
}

/* Card Style */
.main-card {
    background: white;
    padding: 2rem;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

/* Heading */
.title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #6b7280;
    margin-bottom: 1.8rem;
    font-size: 1rem;
}

/* Metric Card */
.result-box {
    background: #f9fafb;
    border-radius: 18px;
    padding: 1.2rem;
    border: 1px solid #e5e7eb;
    margin-top: 1rem;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    padding: 0.75rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
}

/* Labels */
label {
    font-weight: 600 !important;
}

/* Footer */
.footer {
    text-align:center;
    color:#6b7280;
    font-size:0.9rem;
    margin-top:1.5rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = scaler.feature_names_in_

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">💳 Credit Risk Scoring System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Modern AI-powered loan default prediction dashboard built with XGBoost.</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount (₹)", min_value=0.0, value=100000.0, step=5000.0)
    fico = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=720.0)

with col2:
    annual_inc = st.number_input("Annual Income (₹)", min_value=0.0, value=600000.0, step=10000.0)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0, step=0.5)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
if st.button("🔍 Predict Risk"):

    input_dict = {col: 0 for col in feature_names}

    if "loan_amnt" in input_dict:
        input_dict["loan_amnt"] = loan_amnt

    if "annual_inc" in input_dict:
        input_dict["annual_inc"] = annual_inc

    if "fico_range_low" in input_dict:
        input_dict["fico_range_low"] = fico

    if "dti" in input_dict:
        input_dict["dti"] = dti

    input_data = pd.DataFrame([input_dict])[feature_names]

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[:,1][0]

    if prob < 0.30:
        band = "🟢 Low Risk"
    elif prob < 0.60:
        band = "🟠 Medium Risk"
    else:
        band = "🔴 High Risk"

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("Default Probability", f"{prob:.2%}")
    st.write("### Risk Category:", band)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="footer">Built with Python • XGBoost • Streamlit</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)