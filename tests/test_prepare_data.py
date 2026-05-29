import pandas as pd

from src.data.prepare_data import clean_churn_data


def test_clean_churn_data_converts_target_and_total_charges():
    raw = pd.DataFrame(
        {
            "customerID": ["001", "002", "003"],
            "TotalCharges": ["29.85", " ", "108.15"],
            "Churn": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
        }
    )

    cleaned = clean_churn_data(raw, target_column="Churn")

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].dtype.kind in {"f", "i"}
    assert cleaned["Churn"].tolist() == [0, 0]
    assert len(cleaned) == 2


def test_clean_churn_data_rejects_unexpected_target_values():
    raw = pd.DataFrame(
        {
            "TotalCharges": ["29.85"],
            "Churn": ["Maybe"],
        }
    )

    try:
        clean_churn_data(raw, target_column="Churn")
    except ValueError as error:
        assert "contains values other than 'Yes' or 'No'" in str(error)
    else:
        raise AssertionError("Expected ValueError for unexpected target values.")
