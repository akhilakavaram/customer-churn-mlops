from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_sample_customer() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
        ]
    )


def get_latest_run_id(experiment_name: str, tracking_uri: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' was not found. "
            "Run 'python -m src.models.train_model' first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' has no runs. "
            "Run 'python -m src.models.train_model' first."
        )

    return runs[0].info.run_id


def load_latest_model(experiment_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    run_id = get_latest_run_id(experiment_name, tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri), run_id


def load_serving_model(config: dict):
    serving_model_path = PROJECT_ROOT / config["serving_model_path"]

    if not serving_model_path.exists():
        raise FileNotFoundError(
            "Serving model was not found. Run 'python -m src.models.train_model' first."
        )

    model = mlflow.sklearn.load_model(str(serving_model_path))

    try:
        run_id = get_latest_run_id(config["experiment_name"], config["tracking_uri"])
    except ValueError:
        run_id = "unknown"

    return model, run_id


def predict_sample_customer() -> dict:
    config = load_config()
    model, run_id = load_serving_model(config)
    sample_customer = create_sample_customer()

    prediction = int(model.predict(sample_customer)[0])
    probability = float(model.predict_proba(sample_customer)[0][1])

    result = {
        "run_id": run_id,
        "prediction": prediction,
        "churn_probability": probability,
    }

    print(f"Loaded MLflow run ID: {run_id}")
    print(f"Prediction: {prediction}")
    print(f"Churn probability: {probability:.4f}")

    return result


if __name__ == "__main__":
    predict_sample_customer()
