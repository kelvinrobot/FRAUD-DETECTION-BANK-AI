from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from keras.models import load_model
import logging
import os

#  Initialize App 
app = FastAPI(title="AI Fraud Detection API")

#  CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #  my frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Logging 
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#  Thresholds 
DENSE_THRESHOLD = 0.5
AE_THRESHOLD = 0.1
LSTM_THRESHOLD = 0.5

# Load Models and Scalers 
model_dense = load_model("models/fraud_model_dense.h5")
model_auto = load_model("models/fraud_model_autoencoder.h5", compile=False)
model_lstm = load_model("models/fraud_model_lstm.h5")

scaler_dense = joblib.load("scalers/scaler_dense.pkl")
scaler_ae = joblib.load("scalers/scaler_autoencoder.pkl")
scaler_lstm = joblib.load("scalers/scaler_lstm.pkl")

# Request Models 
class SingleTransaction(BaseModel):
    features: List[float]

class BatchTransactions(BaseModel):
    features: List[List[float]]

#  Ensure Python Native Types 
def to_python_type(value):
    if isinstance(value, (np.generic, np.bool_)):
        return value.item()
    return value

# Core Prediction Function
def predict_transaction(features: List[float]) -> dict:
    x = np.array(features).reshape(1, -1)

    # Dense model
    x_dense = scaler_dense.transform(x)
    dense_score = model_dense.predict(x_dense)[0][0]

    # Autoencoder
    x_ae = scaler_ae.transform(x)
    recon = model_auto.predict(x_ae)
    ae_error = np.mean(np.square(x_ae - recon))

    # LSTM
    x_lstm = scaler_lstm.transform(x).reshape(1, 1, 29)
    lstm_score = model_lstm.predict(x_lstm)[0][0]

    # Fraud Logic
    is_fraud = (
        dense_score > DENSE_THRESHOLD or
        ae_error > AE_THRESHOLD or
        lstm_score > LSTM_THRESHOLD
    )

    return {
        "fraud_detected": bool(is_fraud),
        "dense_score": float(dense_score),
        "autoencoder_error": float(ae_error),
        "lstm_score": float(lstm_score)
    }

# Endpoints 
@app.get("/")
def root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_single(data: SingleTransaction):
    if len(data.features) != 29:
        raise HTTPException(status_code=400, detail="Transaction must have exactly 29 features.")
    
    result = predict_transaction(data.features)
    logging.info(f"Single prediction input={data.features}, result={result}")
    return result

@app.post("/predict_batch")
def predict_batch(data: BatchTransactions):
    predictions = []

    for i, features in enumerate(data.features):
        if len(features) != 29:
            predictions.append({
                "index": i,
                "error": "Each transaction must have exactly 29 features."
            })
            continue

        result = predict_transaction(features)
        predictions.append(result)

    logging.info(f"Batch prediction: count={len(data.features)}, results={predictions}")
    return predictions
