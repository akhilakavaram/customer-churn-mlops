from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def clean_churn_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    cleaned = df.copy()

    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    if target_column not in cleaned.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    cleaned[target_column] = cleaned[target_column].map({"No": 0, "Yes": 1})

    if cleaned[target_column].isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains values other than 'Yes' or 'No'."
        )

    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def prepare_data() -> Path:
    config = load_config()

    raw_path = PROJECT_ROOT / config["raw_data_path"]
    processed_path = PROJECT_ROOT / config["processed_data_path"]
    target_column = config["target_column"]

    if not raw_path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Download the Telco churn CSV and place it at "
            f"{raw_path}"
        )

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_churn_data(raw_df, target_column)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)

    print(f"Raw rows: {len(raw_df)}")
    print(f"Processed rows: {len(cleaned_df)}")
    print(f"Processed data saved to: {processed_path}")

    return processed_path


if __name__ == "__main__":
    prepare_data()
