import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---- Load model artifacts ----
model = joblib.load("attrition_model.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_list = joblib.load("feature_list.pkl")

# ---- Streamlit UI Setup ----
st.set_page_config(layout="wide")
st.title("🔍 Employee Attrition Prediction & Explanation")

st.sidebar.header("Enter Employee Details")

# ---- Collect user input ----
def collect_user_input():
    data = {}
    for feat in feature_list:
        if feat in encoders:  # categorical features
            options = encoders[feat].classes_
            data[feat] = st.sidebar.selectbox(f"{feat}", options)
        else:  # numeric features
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
    st.subheader("🔎 Feature Contribution (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Handle binary classification (list) vs single output (scalar)
    if isinstance(explainer.expected_value, list):
        expected_val = explainer.expected_value[1]
        shap_val = shap_values[1]
    else:
        expected_val = explainer.expected_value
        shap_val = shap_values

    # Create SHAP force plot
    fig = shap.force_plot(
        expected_val,
        shap_val,
        input_df,
        matplotlib=True
    )
    st.pyplot(fig)

    # Global SHAP summary
    st.subheader("📊 Global Feature Importance")
    fig_summary = plt.figure()
    shap.summary_plot(shap_val, input_df, show=False)
    st.pyplot(fig_summary)
