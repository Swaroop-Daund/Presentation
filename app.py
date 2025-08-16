# app.py — Streamlit Web App for Employee Attrition Prediction
import streamlit as st
import pandas as pd
import joblib
import shap

# ---- Load model artifacts ----
model = joblib.load("sample_dataattrition_model.pkl")
encoders = joblib.load("sample_datalabel_encoders.pkl")
feature_list = joblib.load("sample_datafeature_list.pkl")

st.set_page_config(layout="wide")
st.title("🔍 Employee Attrition Prediction & Explanation")

st.sidebar.header("Enter Employee Details")

# ---- Input collection ----
def collect_user_input():
    data = {}
    for feat in feature_list:
        if feat in encoders:  # categorical input
            options = encoders[feat].classes_
            data[feat] = st.sidebar.selectbox(f"{feat}", options)
        else:  # numeric input
            data[feat] = st.sidebar.number_input(f"{feat}", value=0.0)
    return pd.DataFrame([data])

input_df = collect_user_input()

# ---- Encode categorical variables ----
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# ---- Prediction ----
if st.sidebar.button("Predict Attrition Risk"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Attrition Risk — {proba:.2%} probability")
    else:
        st.success(f"✅ Low Attrition Risk — {1-proba:.2%} probability")

    # ---- SHAP Explanation ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.subheader("🔎 Feature Contribution (SHAP)")
    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot(
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1],
            input_df,
            matplotlib=True
        )
    )
