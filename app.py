import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("model/model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

app = FastAPI()


class PredictionRequest(BaseModel):
    gender: str
    tenure: int
    MonthlyCharges: float
    Contract: str


@app.post("/predict")
def predict(request: PredictionRequest):
    row = request.model_dump()

    for col, le in preprocessor["label_encoders"].items():
        row[col] = int(le.transform([str(row[col])])[0])

    features = pd.DataFrame([row])[preprocessor["feature_columns"]]
    proba = model.predict_proba(features)[0, 1]

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": "Yes" if proba >= 0.5 else "No",
    }