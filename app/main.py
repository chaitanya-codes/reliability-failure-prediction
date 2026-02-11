import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("./best_model.pkl")

@app.get("/")
def home():
    return {"message": "Failure Prediction API"}

@app.post("/predict")
def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction[0]),
        "failure_probability": float(probability)
    }