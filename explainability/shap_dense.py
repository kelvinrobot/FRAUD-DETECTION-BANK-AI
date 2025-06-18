import shap
import numpy as np
import joblib
from keras.models import load_model
import os

# Define custom feature names
FEATURE_NAMES = [
    "Transaction Amount", "Time Since Last Tx", "User Risk Score", "Device Type Score",
    "IP Reputation", "Card Usage Rate", "Geo Location Score", "Merchant Trust Level",
    "Transaction Velocity", "Weekend Indicator", "Time of Day", "Browser Fingerprint Score",
    "Account Age", "Login Frequency", "Unusual Country Flag", "VPN Usage Score",
    "Previous Fraud Count", "Card Type Score", "Issuer Country Risk", "Failed Login Count",
    "Session Duration", "Clickstream Depth", "Transaction Pattern Deviation",
    "Recent Password Change", "Biometric Verification Score", "Proxy Usage",
    "Multiple Devices Flag", "Previous Disputes", "Behavior Drift Score"
]

# Load model and scaler
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model = load_model(os.path.join(BASE_DIR, "models", "fraud_model_dense.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "scalers", "scaler_dense.pkl"))

# Use background data for SHAP
background = np.zeros((1, model.input_shape[1]))
explainer = shap.Explainer(model, masker=background)

# Explain function
def explain_dense(input_features):
    input_array = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    shap_values = explainer(scaled_input)  # shap.Explanation
    return shap_values

# Exportable symbols
__all__ = ["explain_dense", "FEATURE_NAMES"]
