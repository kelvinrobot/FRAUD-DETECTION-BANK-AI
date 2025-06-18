import numpy as np
import joblib
from keras.models import load_model
import os

# Setup 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "fraud_model_autoencoder.h5")
scaler_path = os.path.join(BASE_DIR, "scalers", "scaler_autoencoder.pkl")

# Load model and scaler
model = load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

#  Explain Function 
def explain_autoencoder(input_features):
    """
    Calculate reconstruction errors from autoencoder model as anomaly signal.
    Returns 1D numpy array with shape (29,) representing error per feature.
    """
    input_array = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    
    # Predict reconstructed version
    reconstructed = model.predict(scaled_input, verbose=0)
    
    # Compute absolute reconstruction error
    reconstruction_error = np.abs(scaled_input - reconstructed).flatten()
    return reconstruction_error
