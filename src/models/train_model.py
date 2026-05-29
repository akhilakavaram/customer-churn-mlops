from pathlib import Path
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    encoder_kwargs = {"handle_unknown": "ignore"}

    try:
        OneHotEncoder(sparse_output=False)
        encoder_kwargs["sparse_output"] = False
    except TypeError:
        encoder_kwargs["sparse"] = False

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(**encoder_kwargs)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_model_pipeline(features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }


def train_model() -> dict:
    config = load_config()
    mlflow.set_tracking_uri(config["tracking_uri"])

    data_path = PROJECT_ROOT / config["processed_data_path"]
    target_column = config["target_column"]

    if not data_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run 'python -m src.data.prepare_data' first."
        )

    df = pd.read_csv(data_path)
    features, target = split_features_and_target(df, target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=target,
    )

    pipeline = build_model_pipeline(x_train)

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "test_size": config["test_size"],
                "random_state": config["random_state"],
                "class_weight": "balanced",
            }
        )

        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_pred_proba = pipeline.predict_proba(x_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=config["registered_model_name"],
        )

        serving_model_path = PROJECT_ROOT / config["serving_model_path"]
        if serving_model_path.exists():
            shutil.rmtree(serving_model_path)
        mlflow.sklearn.save_model(sk_model=pipeline, path=serving_model_path)

        print(f"MLflow run ID: {run.info.run_id}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"Serving model saved to: {serving_model_path}")

        return metrics


if __name__ == "__main__":
    train_model()
