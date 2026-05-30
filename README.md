# Customer Churn MLOps

An end-to-end MLOps learning project for customer churn prediction.

This project is intentionally built in phases. The goal is not only to train a model, but to learn how ML systems are versioned, tested, packaged, deployed, monitored, and retrained.

## Phase 1: Local ML Foundation

Current focus:

- Load the raw Telco customer churn dataset.
- Clean obvious data quality issues.
- Create a processed dataset.
- Prepare the project for experiment tracking and model training.

Coming next:

- Train a baseline model with scikit-learn.
- Track experiments with MLflow.
- Save and load model artifacts.

## Project Structure

```text
customer-churn-mlops/
  configs/              # YAML config files
  data/
    raw/                # Original downloaded datasets
    processed/          # Cleaned/transformed datasets
  notebooks/            # Optional exploration notebooks
  src/
    data/               # Data loading and cleaning code
    features/           # Feature engineering code
    models/             # Training and inference code
  tests/                # Automated tests
```

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Download the Telco Customer Churn dataset and place the CSV here:

```text
data/raw/telco_customer_churn.csv
```

Expected target column:

```text
Churn
```

## First Command

After the CSV is in `data/raw`, run:

```powershell
python -m src.data.prepare_data
```

This creates:

```text
data/processed/churn_cleaned.csv
```

## Train The Baseline Model

After the processed dataset exists, run:

```powershell
python -m src.models.train_model
```

This trains a logistic regression baseline and logs the run to MLflow.

To open the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## Run A Sample Prediction

After at least one MLflow training run exists, run:

```powershell
python -m src.models.predict_model
```

This loads the latest model from the `customer-churn-baseline` experiment and predicts churn for one sample customer.

## Run The API

Start the FastAPI service:

```powershell
uvicorn src.api.main:app --reload
```

Open the interactive docs:

```text
http://127.0.0.1:8000/docs
```

Sample request body for `POST /predict`:

```json
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
  "TotalCharges": 29.85
}
```

## Run With Docker

Export the serving model by running training once:

```powershell
python -m src.models.train_model
```

Build the API image:

```powershell
docker build -t customer-churn-api:local .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 customer-churn-api:local
```

Open:

```text
http://127.0.0.1:8000/docs
```

The Docker image includes the current exported model in `models/customer_churn_pipeline`. It does not need local MLflow tracking files such as `mlflow.db` or `mlruns` to serve predictions. If you retrain the model, rebuild the image so the container includes the latest model.

The Docker build uses `requirements-api.txt`, a smaller Linux-safe dependency file for serving. The full `requirements.txt` is your local development environment and may include Windows-only packages.

## CI/CD

GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The CI job:

- installs Linux-safe development dependencies from `requirements-dev.txt`
- downloads the public Telco churn dataset
- prepares the processed dataset
- trains and exports the serving model
- runs tests
- builds the Docker image

This keeps large/generated artifacts out of Git while still proving the project can rebuild itself from source.

## Learning Notes

In a production MLOps project, raw data is usually treated as immutable. We keep the original dataset in `data/raw` and write cleaned versions to `data/processed` so the transformation is repeatable.

The training script saves a full scikit-learn pipeline, not just a classifier. That matters because production requests will still contain raw categorical fields, and the deployed model must apply the same preprocessing used during training.

The prediction script introduces the inference contract: every prediction request must provide the same feature columns the model saw during training.

The API turns that same contract into HTTP JSON. This is the boundary where model code starts becoming a deployable service.

Docker packages that service with its runtime dependencies. This is the first step toward running the same model API in CI, Kubernetes, or a cloud service.
