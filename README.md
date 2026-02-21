# 🌍 Sri Lanka Tourism Demand Forecasting (Explainable AI)

## 📌 Project Overview

This project forecasts **Sri Lanka’s next-month tourist arrivals** using historical tourism data and economic indicators.  

The system combines:
- 📊 Machine Learning (XGBoost Regressor)
- 🔎 Explainable AI (SHAP)
- 🌐 Streamlit Web Application Deployment

It demonstrates an end-to-end ML pipeline from data preprocessing to deployment.

---

## 🎯 Problem Statement

Accurate tourism demand forecasting is essential for:

- Government policy and planning  
- Hotel and accommodation management  
- Airline scheduling  
- Budget allocation and infrastructure planning  

This system predicts **next month’s total tourist arrivals** using historical trends and macroeconomic data.

---

## 📊 Data Sources

### 1️⃣ Sri Lanka Tourism Development Authority (SLTDA)
- Monthly aggregated tourist arrivals
- Public, non-sensitive data

### 2️⃣ Central Bank of Sri Lanka (CBSL)
- Monthly average USD/LKR exchange rate
- Included to capture economic influence on travel demand

---

## 🧠 Model Used

### XGBoost Regressor

Selected because it:
- Captures nonlinear relationships
- Handles feature interactions
- Performs strongly on structured/tabular data
- Works well with limited datasets

No deep learning models were used, fully complying with assignment constraints.

---

## 🔎 Explainable AI (XAI)

This project integrates **SHAP (SHapley Additive exPlanations)**.

SHAP provides:
- Feature-level contribution for each prediction
- Direction of impact (positive or negative)
- Strength of influence

This ensures the model is transparent and not a black box.

---

## ⚙️ Features Used

- Month
- Quarter
- USD/LKR exchange rate
- Arrivals lag_1 (previous month)
- Arrivals lag_12 (same month last year)
- 3-month rolling mean

