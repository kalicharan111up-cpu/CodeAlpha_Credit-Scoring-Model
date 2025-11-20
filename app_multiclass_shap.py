"""
app_multiclass_shap.py

Streamlit app for multiclass credit scoring with SHAP explainability.
Run: streamlit run app_multiclass_shap.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os, shap, matplotlib.pyplot as plt

st.set_page_config(page_title="💳 Credit Scoring AI Dashboard", layout="wide", page_icon="💳")

# ---------------------------
# Load model & metadata
# ---------------------------
MODEL_PATH = "credit_model.pkl"
META_PATH = "metadata.json"

st.markdown(
    """
    <style>
    .good {background-color:#4CAF50;color:white;padding:15px;border-radius:10px;text-align:center;font-size:24px;}
    .standard {background-color:#FFC107;color:black;padding:15px;border-radius:10px;text-align:center;font-size:24px;}
    .poor {background-color:#F44336;color:white;padding:15px;border-radius:10px;text-align:center;font-size:24px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💳 Credit Scoring Prediction with Explainability")
st.markdown("### Predict and interpret a customer's **Credit Score Category** — `Poor`, `Standard`, or `Good`.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error("❌ Model or metadata not found. Please train using `train_model_multiclass.py` first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(META_PATH, "r") as f:
    meta = json.load(f)

classes = meta["classes"]
num_cols = meta["numeric"]
cat_cols = meta["categorical"]

# ---------------------------
# Sidebar Input Form
# ---------------------------
st.sidebar.header("🧮 Enter Customer Details")

def get_inputs():
    data = {}
    st.sidebar.subheader("Numeric Features")
    for col in num_cols:
        data[col] = st.sidebar.number_input(col, step=0.01, value=0.0)

    if cat_cols:
        st.sidebar.subheader("Categorical Features")
        for col in cat_cols:
            data[col] = st.sidebar.text_input(col, value="")
    return pd.DataFrame([data])

user_df = get_inputs()

# ---------------------------
# Prediction Section
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Prediction Result")

    if st.sidebar.button("🚀 Predict Credit Score"):
        try:
            preds = model.predict(user_df)[0]
            probs = model.predict_proba(user_df)[0]
            prob_df = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values("Probability", ascending=False)

            top_class = prob_df.iloc[0]["Class"]

            # Color-coded card
            if top_class.lower() == "good":
                st.markdown(f"<div class='good'>✅ Credit Score: {top_class}</div>", unsafe_allow_html=True)
            elif top_class.lower() == "standard":
                st.markdown(f"<div class='standard'>⚖️ Credit Score: {top_class}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='poor'>❌ Credit Score: {top_class}</div>", unsafe_allow_html=True)

            # Show probability bars
            st.markdown("### 🔍 Prediction Probabilities")
            for _, row in prob_df.iterrows():
                st.write(f"**{row['Class']}** ({row['Probability']:.2%})")
                st.progress(row["Probability"])

            # ---------------------------
            # SHAP Explainability
            # ---------------------------
            st.markdown("---")
            st.subheader("🧠 Explain Prediction with SHAP")

            try:
                # Get model and preprocessor
                preprocessor = model.named_steps["pre"]
                clf = model.named_steps["clf"]

                X_trans = preprocessor.transform(user_df)
                explainer = shap.Explainer(clf, feature_names=preprocessor.get_feature_names_out())
                shap_values = explainer(X_trans)

                # Identify predicted class index
                pred_idx = list(classes).index(preds)

                # Force Plot / Waterfall
                st.markdown("#### 🔎 Individual Feature Contributions")
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_values[pred_idx][0], show=False)
                st.pyplot(fig)

                # Summary bar
                st.markdown("#### 📈 Global Feature Importance")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_trans, show=False)
                st.pyplot(fig2)

            except Exception as e:
                st.warning(f"⚠️ SHAP explanation not available for this model type: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("👈 Enter inputs and click **Predict Credit Score** to get results and SHAP explanations.")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=220)
    st.markdown("#### Model Information")
    st.json({
        "Trained Model": meta["model"],
        "Target Classes": meta["classes"],
        "Numeric Features": len(meta["numeric"]),
        "Categorical Features": len(meta["categorical"])
    })

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + SHAP | Multiclass Credit Scoring AI App")
