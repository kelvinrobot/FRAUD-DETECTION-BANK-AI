import numpy as np
import joblib
from keras.models import load_model
import os

# Setup 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "fraud_model_lstm.h5")
scaler_path = os.path.join(BASE_DIR, "scalers", "scaler_lstm.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

def explain_lstm(input_features):
    """
    Simulate a synthetic time-series input to LSTM and visualize how feature values
    would evolve. Return difference between first and last timestep (mocked).
    """
    input_array = np.array(input_features).reshape(1, -1)  # (1, 29)
    scaled_input = scaler.transform(input_array)

    # Create synthetic time-series by slightly perturbing features over time
    time_series_input = np.array([
        scaled_input + np.random.normal(0, 0.01 * (i+1), scaled_input.shape)
        for i in range(10)
    ])  # shape (10, 1, 29)

    time_series_input = np.swapaxes(time_series_input, 0, 1)  # shape (1, 10, 29)

    # Get predictions 
    _ = model.predict(time_series_input, verbose=0)

    # Return synthetic delta per feature
    delta = (time_series_input[0, -1, :] - time_series_input[0, 0, :])  # (29,)
    return delta.tolist()
