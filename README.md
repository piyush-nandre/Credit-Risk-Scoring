# Credit-Risk-Scoring


## Overview

This project is a Machine Learning based Credit Risk Scoring application that predicts the probability of loan default using historical lending data.

The solution compares multiple classification models and deploys the best-performing model through a Streamlit web application.

---

## Features

✅ Loan default probability prediction  
✅ Risk categorization (Low / Medium / High)  
✅ Model comparison (3 ML models)  
✅ Explainable AI using SHAP  
✅ Streamlit web app deployment  
✅ Lightweight production-ready model

---

## Models Used

1. Logistic Regression  
2. Decision Tree  
3. XGBoost (Best Model Selected)

---

## Final Model Performance

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | 0.9677 |
| Decision Tree | 0.9613 |
| XGBoost | 0.9835 |

XGBoost was selected as the final deployed model.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit

---

## Project Files

```text
app.py
credit_risk_model.pkl
scaler.pkl
requirements.txt
README.md

---

## Run Locally

pip install -r requirements.txt
streamlit run app.py

---

## Web Deployment

Hosted using Streamlit Community Cloud.

---


##  Input Features (Demo App)

Loan Amount
Annual Income
Credit Score

---

## Output

Default Probability
Risk Category
Learning Outcomes
Credit Risk Analytics
FinTech Machine Learning
Model Evaluation
Explainable AI
Web App Deployment

---

## Author

Piyush Nandre
Completed as part of a FinTech Analytics project series.
