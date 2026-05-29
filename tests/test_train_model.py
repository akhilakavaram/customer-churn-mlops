import pandas as pd

from src.models.train_model import build_model_pipeline, calculate_metrics, split_features_and_target


def test_split_features_and_target_separates_target_column():
    df = pd.DataFrame(
        {
            "tenure": [1, 12],
            "Contract": ["Month-to-month", "One year"],
            "Churn": [1, 0],
        }
    )

    features, target = split_features_and_target(df, target_column="Churn")

    assert features.columns.tolist() == ["tenure", "Contract"]
    assert target.tolist() == [1, 0]


def test_build_model_pipeline_can_fit_mixed_feature_types():
    features = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 3],
            "MonthlyCharges": [29.85, 56.95, 89.1, 42.3],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        }
    )
    target = pd.Series([1, 0, 0, 1])

    pipeline = build_model_pipeline(features)
    pipeline.fit(features, target)

    predictions = pipeline.predict(features)

    assert len(predictions) == len(target)


def test_calculate_metrics_returns_expected_metric_names():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0])
    y_pred_proba = pd.Series([0.1, 0.8, 0.2, 0.4])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
