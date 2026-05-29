from fastapi.testclient import TestClient

from src.api.main import app, get_model


class FakeModel:
    def predict(self, customer_df):
        return [1]

    def predict_proba(self, customer_df):
        return [[0.2, 0.8]]


def sample_payload() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_health_check_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_response():
    app.dependency_overrides[get_model] = lambda: (FakeModel(), "test-run-id")
    client = TestClient(app)

    response = client.post("/predict", json=sample_payload())

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "churn_probability": 0.8,
        "model_run_id": "test-run-id",
    }
