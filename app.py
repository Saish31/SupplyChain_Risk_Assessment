import streamlit as st
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Scaler & Encoder ─────────────────────────────────────────────────────
model_dir = "models"
# Scaler trained on only the 3 input features: 'On_Time_Delivery_Rate', 'Financial_Stability_Score', 'Severity'
scaler = joblib.load(os.path.join(model_dir, "scaler_input_only.pkl"))
# LabelEncoder for Severity ('Low', 'Medium', 'High')
severity_encoder = joblib.load(os.path.join(model_dir, "severity_encoder.pkl"))

# ── Load Random Forest Models (3 features each) ───────────────────────────────
targets = [
    "Penalty_Cost_USD",
    "Compensation_Paid_USD",
    "Disruption_Cost_USD",
    "Time_To_Recovery_Days"
]
rf_models = {}
for target in targets:
    key = target.replace(" ", "_")
    model_path = os.path.join(model_dir, f"rf_{key}_3feat.pkl")
    rf_models[target] = joblib.load(model_path)

# ── Cached SHAP Explainer ─────────────────────────────────────────────────────
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.title("📋 Input Parameters")
on_time_delivery    = st.sidebar.slider("On-Time Delivery Rate",       0.0, 1.0, 0.85)
financial_stability = st.sidebar.slider("Financial Stability Score",   0.0, 1.0, 0.75)
severity            = st.sidebar.selectbox("Severity", ["Low", "Medium", "High"])
show_shap           = st.sidebar.checkbox("Show SHAP Explanations")
if show_shap:
    target_explain = st.sidebar.selectbox("Select KPI to explain", targets)

# Encode & scale
severity_encoded = severity_encoder.transform([severity])[0]
X = np.array([[on_time_delivery, financial_stability, severity_encoded]])
X_scaled = scaler.transform(X)

# ── Main Title ────────────────────────────────────────────────────────────────
st.title("📊 Supply Chain Risk Prediction Dashboard")

# ── Predictions ───────────────────────────────────────────────────────────────
st.subheader("🔮 Predictions")
preds = {t: rf_models[t].predict(X_scaled)[0] for t in targets}

c1, c2 = st.columns(2)
with c1:
    st.metric("Penalty Cost (USD)"     , f"${preds['Penalty_Cost_USD']:,.2f}")
    st.metric("Disruption Cost (USD)"  , f"${preds['Disruption_Cost_USD']:,.2f}")
with c2:
    st.metric("Compensation Paid (USD)", f"${preds['Compensation_Paid_USD']:,.2f}")
    st.metric("Time to Recovery (days)", f"{preds['Time_To_Recovery_Days']:.1f} days")

# ── Risk Summary & Recommendation ────────────────────────────────────────────
st.subheader("🧠 Risk Summary & Recommendation")
risk_level = (
    "High"     if severity == "High"
    else "Moderate" if severity == "Medium"
    else "Low"
)
if risk_level == "High":
    rec = "🚨 **Consider alternate sourcing.** High severity suggests a significant disruption risk."
elif risk_level == "Moderate":
    rec = "⚠️ **Monitor this supplier closely.** Medium severity indicates potential delays or moderate risks."
else:
    rec = "✅ **Supplier risk is low.** Continue operations as planned."

st.markdown(f"**Risk Level:** {risk_level}")
st.info(rec)

# ── SHAP Explanation ──────────────────────────────────────────────────────────
if show_shap:
    st.subheader(f"🔍 SHAP Explanation for {target_explain}")
    explainer = get_explainer(rf_models[target_explain])
    shap_values = explainer.shap_values(X_scaled)
    feature_names = ["On_Time_Delivery_Rate", "Financial_Stability_Score", "Severity"]
    # Bar chart of local SHAP values
    fig, ax = plt.subplots()
    ax.bar(feature_names, shap_values[0])
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Local SHAP values for {target_explain}")
    ax.axhline(0, linewidth=0.8)
    st.markdown("""
    **What does this plot mean?**

    This bar chart shows how each input feature contributed to the model's prediction for the selected KPI.  
    - Positive SHAP values (bars above zero) **increase** the predicted value.  
    - Negative SHAP values (bars below zero) **decrease** the predicted value.  
    - The magnitude of each bar shows the **influence** of that feature on the prediction.

    For example, if `Severity` has a high positive SHAP value, it means it significantly increased the predicted cost or delay.
    """)

    st.pyplot(fig)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | ML Model: Random Forest (3 Features) with SHAP")
