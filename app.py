from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import joblib
from custom_model import LogisticRegression

app = FastAPI(title="Breast Cancer Prediction API")

# Load trained model parameters
with open("cancer_model_params.json", "r") as f:
    params = json.load(f)

model = LogisticRegression(
    learning_rate=params["learning_rate"],
    n_iterations=params["n_iterations"]
)
model.weights = np.array(params["weights"])
model.bias = params["bias"]

# Load the SAME scaler used during training
scaler = joblib.load("cancer_scaler.joblib")


class PatientData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running"}


@app.post("/predict")
def predict(data: PatientData):
    X = np.array(data.features, dtype=float).reshape(1, -1)

    if X.shape[1] != len(model.weights):
        return {
            "error": f"Expected {len(model.weights)} features, received {X.shape[1]}"
        }

    # Scale raw input before prediction
    X_scaled = scaler.transform(X)

    probability = float(model.predict_proba(X_scaled)[0])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "class_1_probability": probability
    }