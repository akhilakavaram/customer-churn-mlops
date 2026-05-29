from functools import lru_cache

import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.models.predict_model import load_config, load_serving_model


app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
)


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
    model_run_id: str


@lru_cache(maxsize=1)
def get_model():
    config = load_config()
    model, run_id = load_serving_model(config)
    return model, run_id


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures, model_context=Depends(get_model)) -> PredictionResponse:
    model, run_id = model_context
    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(customer_df)[0])
    churn_probability = float(model.predict_proba(customer_df)[0][1])

    return PredictionResponse(
        prediction=prediction,
        churn_probability=churn_probability,
        model_run_id=run_id,
    )
