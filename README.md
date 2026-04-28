# 💳 Credit Risk Scoring System

An AI-powered credit risk assessment web application that predicts the probability of loan default using Machine Learning and Explainable AI.

Built using Python, Streamlit, XGBoost, and SHAP.

---

## 📌 Project Overview

Financial institutions need accurate risk evaluation before approving loans. This project predicts whether an applicant is likely to default based on financial and credit-related attributes.

The system helps simulate real-world underwriting decisions by classifying applicants into:

- 🟢 Low Risk
- 🟠 Medium Risk
- 🔴 High Risk

---

## 🚀 Version 4 Improvements

Compared to previous versions, V4 includes:

- Improved feature engineering
- Cleaner preprocessing pipeline
- End-to-end Scikit-learn Pipeline
- XGBoost classifier
- Better generalization
- SHAP Explainability
- Compare Applicants feature
- Professional Streamlit dashboard

---

## 🧠 Model Details

**Algorithm Used:** XGBoost Classifier  
**Pipeline Used:** Preprocessing + Modeling in one pipeline

### Key Features Used

- Loan Amount
- Annual Income
- Credit Score
- Debt-to-Income Ratio
- Interest Rate
- Open Accounts
- Total Accounts
- Credit Utilization
- Delinquencies
- Public Records
- Employment Length
- Home Ownership
- Loan Purpose

---

## 📊 Performance

Version 4 focused on honest, non-leakage modeling and realistic risk prediction.

**Best Metric:** ROC-AUC ~0.74 (realistic production-style evaluation)

---

## 🖥️ Application Features

- Predict Default Probability
- Risk Category Classification
- Loan Approval Recommendation
- Compare Two Applicants
- Premium Dashboard UI
- Explainable AI Insights (SHAP)

---

## 📂 Project Files

```text
app.py
credit_risk_pipeline_v4.pkl
requirements.txt
README.md
