import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Credit Risk Scoring Pro",
    page_icon="💳",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_pipeline_v4.pkl")

model = load_model()

# ==========================================================
# CSS PREMIUM UI
# ==========================================================
st.markdown("""
<style>
header, footer, #MainMenu {visibility:hidden;}
[data-testid="stHeader"] {display:none;}

.stApp{
background: linear-gradient(135deg,#f8fafc,#eef2ff);
}

.block-container{
padding-top:1rem;
max-width:1400px;
}

.main-card{
background:white;
padding:2rem;
border-radius:22px;
box-shadow:0 10px 30px rgba(0,0,0,0.08);
border:1px solid #e5e7eb;
margin-bottom:1rem;
}

.metric-card{
background:#f9fafb;
padding:1rem;
border-radius:18px;
border:1px solid #e5e7eb;
}

.stButton > button{
width:100%;
background:linear-gradient(90deg,#2563eb,#4f46e5);
color:white;
font-weight:700;
border:none;
border-radius:12px;
padding:0.8rem;
}

.stDownloadButton > button{
width:100%;
border-radius:12px;
font-weight:700;
}

.small{
color:#6b7280;
font-size:0.95rem;
}

h1,h2,h3{
color:#111827;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HELPERS
# ==========================================================
def risk_band(prob):
    if prob < 0.30:
        return "🟢 Low Risk"
    elif prob < 0.60:
        return "🟠 Medium Risk"
    return "🔴 High Risk"

def approval(prob):
    if prob < 0.30:
        return "✅ Recommended for Approval"
    elif prob < 0.60:
        return "⚠️ Conditional Approval / Review Needed"
    return "❌ High Risk - Manual Review Required"

def build_input(
    loan_amnt,
    annual_inc,
    fico,
    dti,
    int_rate,
    term,
    open_acc,
    total_acc,
    revol_util,
    inq_last_6mths,
    delinq_2yrs,
    pub_rec,
    purpose,
    home_ownership,
    verification_status,
    emp_length,
    addr_state
):
    return pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": loan_amnt/36,
        "grade": "B",
        "sub_grade": "B3",
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "fico_range_low": fico,
        "fico_range_high": fico + 5,
        "dti": dti,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "revol_util": revol_util,
        "inq_last_6mths": inq_last_6mths,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "purpose": purpose,
        "addr_state": addr_state,
        "fico_mean": (fico + fico + 5)/2,
        "loan_income_ratio": loan_amnt/(annual_inc+1),
        "installment_income_ratio": (loan_amnt/36)/(annual_inc+1),
        "util_x_dti": revol_util*dti
    }])

# ==========================================================
# HEADER
# ==========================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("💳 Credit Risk Scoring Pro")
st.markdown(
    '<div class="small">AI-powered loan default prediction with Explainable AI, comparison engine and downloadable report.</div>',
    unsafe_allow_html=True
)

tabs = st.tabs([
    "📈 Single Applicant",
    "⚖️ Compare Applicants",
    "ℹ️ About Model"
])

# ==========================================================
# TAB 1
# ==========================================================
with tabs[0]:

    c1, c2, c3 = st.columns(3)

    with c1:
        loan_amnt = st.number_input("Loan Amount (₹)", 0.0, value=100000.0, step=5000.0)
        annual_inc = st.number_input("Annual Income (₹)", 0.0, value=600000.0, step=10000.0)
        fico = st.number_input("Credit Score", 300.0, 900.0, value=720.0)

    with c2:
        dti = st.number_input("Debt-to-Income Ratio", 0.0, value=15.0, step=0.5)
        int_rate = st.number_input("Interest Rate (%)", 0.0, value=12.0, step=0.1)
        term = st.selectbox("Term", ["36 months", "60 months"])

    with c3:
        open_acc = st.number_input("Open Accounts", 0, value=5)
        total_acc = st.number_input("Total Accounts", 0, value=10)
        revol_util = st.number_input("Credit Utilization (%)", 0.0, value=30.0)

    c4, c5, c6 = st.columns(3)

    with c4:
        inq_last_6mths = st.number_input("Recent Inquiries", 0, value=1)
        delinq_2yrs = st.number_input("Delinquencies", 0, value=0)

    with c5:
        pub_rec = st.number_input("Public Records", 0, value=0)
        purpose = st.selectbox("Purpose", ["debt_consolidation","credit_card","home_improvement","major_purchase"])

    with c6:
        home_ownership = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
        verification_status = st.selectbox("Verification", ["Verified","Source Verified","Not Verified"])
        emp_length = st.selectbox("Employment Length", ["10+ years","5 years","2 years","< 1 year"])
        addr_state = st.selectbox("State", ["CA","TX","NY","FL","IL","OH","PA"])

    if st.button("🔍 Predict Risk"):

        X = build_input(
            loan_amnt, annual_inc, fico, dti, int_rate, term,
            open_acc, total_acc, revol_util, inq_last_6mths,
            delinq_2yrs, pub_rec, purpose, home_ownership,
            verification_status, emp_length, addr_state
        )

        prob = model.predict_proba(X)[:,1][0]
        band = risk_band(prob)
        reco = approval(prob)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Default Probability", f"{prob:.2%}")

        with m2:
            st.metric("Risk Band", band)

        with m3:
            st.metric("Decision", reco)

        st.progress(float(prob))

        # ==================================================
        # SHAP EXPLAINABILITY
        # ==================================================
        st.subheader("📊 Why This Prediction Happened")

        try:
            booster = model.named_steps["model"]
            prep = model.named_steps["prep"]

            X_t = prep.transform(X)
            feat_names = prep.get_feature_names_out()

            explainer = shap.TreeExplainer(booster)
            vals = explainer.shap_values(X_t)[0]

            shap_df = pd.DataFrame({
                "Feature": feat_names,
                "Impact": vals
            })

            shap_df["Abs"] = shap_df["Impact"].abs()
            shap_df = shap_df.sort_values("Abs", ascending=False).head(10)

            for _, row in shap_df.iterrows():
                txt = "⬆️ Increased Risk" if row["Impact"] > 0 else "⬇️ Reduced Risk"
                st.write(f"**{row['Feature']}** : {txt}")

            fig, ax = plt.subplots(figsize=(9,4))
            colors = ["red" if x > 0 else "green" for x in shap_df["Impact"]]
            ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
            ax.invert_yaxis()
            ax.set_title("Top Factors Affecting Decision")
            st.pyplot(fig)

        except:
            st.info("SHAP explanation unavailable for current environment.")

        # ==================================================
        # DOWNLOAD REPORT
        # ==================================================
        report = pd.DataFrame({
            "Metric": [
                "Default Probability",
                "Risk Band",
                "Decision"
            ],
            "Value": [
                round(prob,4),
                band,
                reco
            ]
        })

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📄 Download Prediction Report (CSV)",
            data=csv,
            file_name="credit_risk_report.csv",
            mime="text/csv"
        )

# ==========================================================
# TAB 2 COMPARE
# ==========================================================
with tabs[1]:

    st.subheader("⚖️ Compare Two Applicants")

    a1, a2 = st.columns(2)

    with a1:
        loan1 = st.number_input("Applicant A Loan", 0.0, value=100000.0, key="a1")
        fico1 = st.number_input("Applicant A Score", 300.0, 900.0, value=760.0, key="a2")

    with a2:
        loan2 = st.number_input("Applicant B Loan", 0.0, value=200000.0, key="b1")
        fico2 = st.number_input("Applicant B Score", 300.0, 900.0, value=640.0, key="b2")

    if st.button("Compare Applicants"):

        X1 = build_input(
            loan1,600000,fico1,15,12,"36 months",5,10,30,1,0,0,
            "credit_card","RENT","Verified","5 years","CA"
        )

        X2 = build_input(
            loan2,600000,fico2,15,12,"36 months",5,10,30,1,0,0,
            "credit_card","RENT","Verified","5 years","CA"
        )

        p1 = model.predict_proba(X1)[:,1][0]
        p2 = model.predict_proba(X2)[:,1][0]

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Applicant A Risk", f"{p1:.2%}")

        with c2:
            st.metric("Applicant B Risk", f"{p2:.2%}")

        if p1 < p2:
            st.success("Applicant A is lower risk.")
        else:
            st.warning("Applicant B is lower risk.")

# ==========================================================
# TAB 3
# ==========================================================
with tabs[2]:

    st.subheader("ℹ️ About This Project")

    st.write("""
    **Model Used:** XGBoost Pipeline (V4)

    **Purpose:** Predict likelihood of loan default.

    **Features Included:**
    - Credit score
    - Income
    - Debt ratio
    - Utilization
    - Loan amount
    - Term
    - Account behavior

    **Explainable AI:** SHAP-based factor analysis.
    """)

st.markdown('</div>', unsafe_allow_html=True)