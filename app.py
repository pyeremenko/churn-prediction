from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model/model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = joblib.load(MODEL_PATH)
    _state["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    yield
    _state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn based on their account and service attributes",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    gender: str = Field(..., description="Customer gender", examples=["Male", "Female"])
    tenure: int = Field(..., description="Number of months the customer has been with the company", examples=[12])
    MonthlyCharges: float = Field(..., description="The amount charged monthly in USD", examples=[45.3])
    Contract: str = Field(..., description="The contract term of the customer", examples=["Month-tomonth", "One year", "Two year"])


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability that the customer will churn (0.0 - 1.0)")
    prediction: str = Field(..., description="Prediction label: Yes or No")


def _transform(request: PredictionRequest) -> pd.DataFrame:
    preprocessor = _state["preprocessor"]
    label_encoders: dict = preprocessor["label_encoders"]
    feature_columns: list = preprocessor["feature_columns"]

    row = request.model_dump()

    for feature_name, encoder in label_encoders.items():
        value = str(row[feature_name])
        if value not in encoder.classes_:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown value '{value}' for field '{feature_name}'. Expected one of: {list(encoder.classes_)}",
            )
        row[feature_name] = int(encoder.transform([value])[0])

    return pd.DataFrame([row])[feature_columns]


@app.post("/predict", response_model=PredictionResponse, summary="Predict customer churn")
def predict(request: PredictionRequest) -> PredictionResponse:
    features = _transform(request)
    churn_probability = _state["model"].predict_proba(features)[0, 1]
    label = "Yes" if churn_probability >= 0.5 else "No"
    return PredictionResponse(churn_probability=round(float(churn_probability), 4), prediction=label)
