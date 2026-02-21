import streamlit as st
import pandas as pd
import joblib, json
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="SL Tourism Forecast", page_icon="🌍", layout="wide")

# ---------------------------
# Tourism-themed styling
# ---------------------------
CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #d9f2ff, #f0fbff);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.kpi-card {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    border: 1px solid rgba(0,0,0,0.08);
    background: white;
    box-shadow: 0 8px 25px rgba(0, 105, 148, 0.14);
}
h1, h2, h3 {
    color: #005f73;
}
.stButton > button {
    background-color: #0a9396;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #007f86;
}
.stTabs [data-baseweb="tab"] {
    background-color: #e0f7fa;
    border-radius: 10px;
    padding: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #94d2bd !important;
    color: black !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# Load model and features
# ---------------------------
model = joblib.load("tourism_xgb_model.joblib")
feature_cols = json.load(open("feature_cols.json"))

# ---------------------------
# Header
# ---------------------------
st.markdown("""
# 🌍 Sri Lanka Tourism Demand Forecasting  
### 🌊 Machine Learning + Explainable AI (SHAP)
""")
st.markdown("Predict next month's tourist arrivals using historical patterns and exchange rate data.")
st.divider()

# ---------------------------
# Tabs
# ---------------------------
tab_pred, tab_xai, tab_about = st.tabs(["🔮 Predict", "🔎 Explain (XAI)", "ℹ️ About"])

# ---------------------------
# Predict Tab
# ---------------------------
with tab_pred:

    st.subheader("Input Features")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        month_map = {
            "January":1,"February":2,"March":3,"April":4,
            "May":5,"June":6,"July":7,"August":8,
            "September":9,"October":10,"November":11,"December":12
        }

        month_name = st.selectbox("Month", list(month_map.keys()))
        month = month_map[month_name]

        quarter = st.selectbox("Quarter", [1,2,3,4])

        usd_lkr_avg = st.number_input("USD/LKR Exchange Rate", value=300.0)
        arrivals_lag_1 = st.number_input("Arrivals Last Month (lag_1)", value=100000.0)
        arrivals_lag_12 = st.number_input("Arrivals Same Month Last Year (lag_12)", value=120000.0)
        arrivals_roll_3 = st.number_input("3-Month Rolling Average", value=110000.0)

        predict_btn = st.button("🔮 Predict", type="primary")

    with col2:
        st.markdown("""
        <div class="kpi-card">
        <b>Tips:</b><br>
        • Increase lag_12 to simulate seasonal peaks.<br>
        • Rolling mean reflects short-term trend.<br>
        • Exchange rate captures economic influence.
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "month": month,
            "quarter": quarter,
            "usd_lkr_avg": usd_lkr_avg,
            "arrivals_lag_1": arrivals_lag_1,
            "arrivals_lag_12": arrivals_lag_12,
            "arrivals_roll_3": arrivals_roll_3
        }])[feature_cols]

        prediction = model.predict(input_df)[0]

        st.divider()

        st.markdown(
            f"""
            <div class="kpi-card">
            <div style="color:gray;">Predicted Next-Month Tourist Arrivals</div>
            <div style="font-size:2.2rem;font-weight:800;">{prediction:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Save for XAI tab
        st.session_state["last_input"] = input_df
        st.session_state["last_pred"] = prediction


# ---------------------------
# XAI Tab
# ---------------------------
with tab_xai:

    st.subheader("Explainable AI (SHAP)")

    if "last_input" not in st.session_state:
        st.info("Run a prediction first to see explanation.")
    else:
        input_df = st.session_state["last_input"]
        prediction = st.session_state["last_pred"]

        st.markdown(
            f"""
            <div class="kpi-card">
            <b>Latest Prediction:</b> {prediction:,.0f} arrivals
            </div>
            """,
            unsafe_allow_html=True
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Impact": shap_values[0]
        }).sort_values("SHAP Impact", key=abs, ascending=False)

        st.markdown("### Feature Contribution")
        st.dataframe(shap_df)

        st.markdown("### Waterfall Explanation")

        exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        )

        fig = plt.figure()
        shap.waterfall_plot(exp, show=False)
        st.pyplot(fig, clear_figure=True)

        st.markdown("""
        **Interpretation:**  
        Positive SHAP → increases prediction  
        Negative SHAP → decreases prediction  
        Larger value → stronger influence
        """)

# ---------------------------
# About Tab
# ---------------------------
with tab_about:

    st.subheader("About this System")

    st.markdown("""
### 🎯 Problem
Forecast Sri Lanka’s next-month tourist arrivals to support tourism planning and policy decisions.

### 📊 Data
• SLTDA – Monthly total tourist arrivals  
• CBSL – Monthly average USD/LKR exchange rate  

### 🤖 Model
XGBoost Regressor (non-deep learning, tabular ML).  
Captures seasonality and complex feature interactions.

### 🔎 Explainable AI
SHAP explains each prediction by showing feature-level impact, ensuring transparency.
""")

st.divider()
st.caption("Academic ML Assignment — Streamlit Front-End with Explainable AI - 214202P")