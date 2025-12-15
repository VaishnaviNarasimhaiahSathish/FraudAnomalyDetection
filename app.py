import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
pd.set_option("styler.render.max_elements", 10_000_000)


# ============================================================
# 1. AUTOENCODER MODELS (must match your training architecture)
# ============================================================

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# 2. LOAD MODELS (Sklearn + Autoencoders)
# ============================================================

def load_ml_models(model_dir="models"):
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            models[model_name] = joblib.load(os.path.join(model_dir, file))
    return models


def load_autoencoders(input_dim, model_dir="models"):
    ae_path = os.path.join(model_dir, "autoencoder_model.pth")
    deep_path = os.path.join(model_dir, "deep_autoencoder_model.pth")

    ae = SimpleAutoencoder(input_dim)
    deep = DeepAutoencoder(input_dim)

    ae.load_state_dict(torch.load(ae_path, map_location="cpu"))
    deep.load_state_dict(torch.load(deep_path, map_location="cpu"))

    ae.eval()
    deep.eval()
    return ae, deep


# ============================================================
# 3. SIDEBAR UI
# ============================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("Select a model for fraud prediction:")

# Load supervised models (LogReg, RF, XGB)
ml_models = load_ml_models()

# Placeholder list until we know feature count
model_options = list(ml_models.keys()) + ["autoencoder", "deep_autoencoder"]

selected_model = st.sidebar.selectbox("Choose Model", model_options)

# Model descriptions
descriptions = {
    "log_reg_model": (
        "Logistic Regression with class weighting.\n\n"
        "A fast linear baseline model that performs well on high-dimensional fraud data. "
        "It is interpretable and provides calibrated probabilities. "
        "Useful for understanding which features contribute most to fraud likelihood."
    ),

    "rf_model": (
        "Random Forest Classifier.\n\n"
        "An ensemble of decision trees trained with bootstrap aggregation (bagging). "
        "It reduces overfitting and captures non-linear relationships. "
        "Good for detecting unusual transaction patterns and ranking feature importance."
    ),

    "xgb_model": (
        "XGBoost Gradient Boosting Model.\n\n"
        "A highly optimized boosting algorithm known for state-of-the-art performance in fraud detection. "
        "Handles extreme class imbalance well, captures complex interactions, and generally yields "
        "the strongest supervised model performance."
    ),

    "autoencoder": (
        "Simple Autoencoder (Unsupervised Anomaly Detector).\n\n"
        "Learns to reconstruct normal transactions. "
        "Fraudulent behavior appears as high reconstruction error, making it ideal for scenarios "
        "where labels are scarce or fraud patterns evolve over time."
    ),

    "deep_autoencoder": (
        'Deep Autoencoder (Hierarchical Representation Learner).\n\n'
        "With a deeper encoder–decoder structure, this model captures more abstract patterns "
        "in transactional data. It improves separation between normal and anomalous transactions "
        "and is better suited for highly complex fraud behaviors."
    )
}


st.sidebar.markdown("---")
st.sidebar.subheader("Model Description")
st.sidebar.info(descriptions.get(selected_model, "Choose a model from above."))
st.sidebar.markdown("---")
st.sidebar.caption("💳 Fraud Detection Dashboard")


# ============================================================
# 4. MAIN UI
# ============================================================

st.title("💳 Fraud Detection Dashboard")
st.write("Upload a CSV file and detect fraudulent transactions using multiple ML models.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    # Load CSV
    df = pd.read_csv(uploaded_file)

    # Remove target column if included
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Remove unnamed/index columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Detect feature size
    input_dim = df.shape[1]

    # Load AE models WITH CORRECT DIMENSION
    ae_model, deep_ae_model = load_autoencoders(input_dim=input_dim)

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Convert to tensor
    X_tensor = torch.tensor(df.values, dtype=torch.float32)

    # ============================================================
    # 5. MODEL PREDICTION
    # ============================================================

    if selected_model in ml_models:
        model = ml_models[selected_model]

        # Try predict_proba first
        try:
            probs = model.predict_proba(df)[:, 1]
        except:
            # Some models output hard labels
            probs = model.predict(df)

        preds = (probs > 0.5).astype(int)

        result_df = df.copy()
        result_df["Fraud_Probability"] = probs
        result_df["Prediction"] = preds

    elif selected_model == "autoencoder":
        with torch.no_grad():
            reconstructed = ae_model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

        threshold = 0.487
        preds = (errors > threshold).astype(int)

        result_df = df.copy()
        result_df["Reconstruction_Error"] = errors
        result_df["Prediction"] = preds

    elif selected_model == "deep_autoencoder":
        with torch.no_grad():
            reconstructed = deep_ae_model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

        threshold = 0.487
        preds = (errors > threshold).astype(int)

        result_df = df.copy()
        result_df["Reconstruction_Error"] = errors
        result_df["Prediction"] = preds

     # ============================================================
    # 6. VISUAL OUTPUT
    # ============================================================

    st.subheader("Predictions")

    def highlight_fraud(row):
        return ["background-color: #ffcccc" if row["Prediction"] == 1 else "" for _ in row]

    st.dataframe(
        result_df.style.apply(highlight_fraud, axis=1),
        use_container_width=True
    )

    # ===============================
    # Fraud Probability Meter
    # ===============================
    st.markdown("### Fraud Probability Meter")

    avg_fraud = result_df["Prediction"].mean()
    fraud_pct = avg_fraud * 100

    # Determine color for severity badge
    if fraud_pct > 10:
        color = "#ff4d4d"   # red
    elif fraud_pct > 1:
        color = "#ffa64d"   # orange
    else:
        color = "#5cd65c"   # green

    # Show colored severity badge
    st.markdown(
        f"""
        <div style='padding:15px; border-radius:10px; background:{color};
                    color:white; text-align:center; margin-bottom:10px;'>
            <h3 style="margin:0;">{fraud_pct:.2f}% Fraud Detected</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metric for number of detected fraud cases
    st.metric(
        label="Fraudulent Transactions Detected",
        value=f"{result_df['Prediction'].sum()} cases"
    )

    # ===============================
    # DOWNLOAD RESULTS
    # ===============================

    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        mime="text/csv",
        file_name=f"{selected_model}_predictions.csv"
    )

st.markdown("---")
st.caption("Built using Streamlit • Machine Learning • Autoencoders")
