import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

# --------------------------------------------------
# CLEAN PROFESSIONAL UI CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit chrome */
header, footer, #MainMenu {
    visibility: hidden;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
}

/* Main container */
.block-container {
    max-width: 950px;
    margin-top: 1.2rem;
    padding: 2.2rem 2.2rem 1.2rem 2.2rem;
    background: white;
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

/* Headings */
h1 {
    color: #111827;
    font-weight: 800;
    margin-bottom: 0.25rem;
}

.subtext {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 1.6rem;
}

/* Labels */
label {
    font-weight: 600 !important;
    color: #111827 !important;
}

/* Inputs */
.stNumberInput input {
    border-radius: 12px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    padding: 0.8rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
}

/* Cards */
.result-card {
    margin-top: 1rem;
    padding: 1.2rem;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.92rem;
    margin-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = scaler.feature_names_in_

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("💳 Credit Risk Scoring System")
st.markdown(
    '<div class="subtext">Modern AI-powered loan default prediction dashboard built with XGBoost.</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# INPUTS (UPGRADED 10 FEATURES)
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount (₹)", min_value=0.0, value=100000.0, step=5000.0
    )

    fico = st.number_input(
        "Credit Score", min_value=300.0, max_value=900.0, value=720.0
    )

    int_rate = st.number_input(
        "Interest Rate (%)", min_value=0.0, value=12.0, step=0.1
    )

    open_acc = st.number_input(
        "Open Credit Accounts", min_value=0, value=5, step=1
    )

    emp_length = st.selectbox(
        "Employment Length",
        ["<1 year", "1 year", "2 years", "3 years", "5 years", "7 years", "10+ years"]
    )

with col2:
    annual_inc = st.number_input(
        "Annual Income (₹)", min_value=0.0, value=600000.0, step=10000.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio", min_value=0.0, value=15.0, step=0.5
    )

    revol_util = st.number_input(
        "Credit Utilization (%)", min_value=0.0, value=30.0, step=1.0
    )

    inq_last_6mths = st.number_input(
        "Recent Inquiries (6 Months)", min_value=0, value=1, step=1
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE"]
    )
# --------------------------------------------------
# PREDICT BUTTON
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

    if "int_rate" in input_dict:
        input_dict["int_rate"] = int_rate

    if "open_acc" in input_dict:
        input_dict["open_acc"] = open_acc

    if "revol_util" in input_dict:
        input_dict["revol_util"] = revol_util

    if "inq_last_6mths" in input_dict:
        input_dict["inq_last_6mths"] = inq_last_6mths

    if "emp_length" in input_dict:
        input_dict["emp_length"] = emp_length

    if "home_ownership" in input_dict:
        input_dict["home_ownership"] = home_ownership

    input_data = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_data)

    # Prediction
    prob = model.predict_proba(input_scaled)[:, 1][0]

    if prob < 0.30:
        band = "🟢 Low Risk"
    elif prob < 0.60:
        band = "🟠 Medium Risk"
    else:
        band = "🔴 High Risk"

    # Output Card
    
    with st.container():
        st.metric("Default Probability", f"{prob:.2%}")
        st.markdown(f"### {band}")
        st.caption("Predicted Risk Category")

    # --------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------
    st.markdown("## 📊 Why This Prediction Happened")

    shap_values = explainer.shap_values(input_scaled)

    vals = shap_values[0]

    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": vals
    })

    impact_df["AbsImpact"] = impact_df["Impact"].abs()
    impact_df = impact_df.sort_values("AbsImpact", ascending=False).head(8)

    # Text Explanation
    for _, row in impact_df.iterrows():
        direction = "⬆️ Increased Risk" if row["Impact"] > 0 else "⬇️ Reduced Risk"
        st.write(f"**{row['Feature']}** : {direction}")

    # Chart
    fig, ax = plt.subplots(figsize=(8,4))

    colors = ["red" if x > 0 else "green" for x in impact_df["Impact"]]

    ax.barh(impact_df["Feature"], impact_df["Impact"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Top Factors Affecting Prediction")

    st.pyplot(fig)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="footer">Built with Python • XGBoost • Streamlit • SHAP AI</div>',
    unsafe_allow_html=True
)