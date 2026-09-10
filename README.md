# Food Delivery ETA Prediction

A machine learning project that predicts food delivery time for Swiggy-style delivery data, built as a reusable end-to-end ML pipeline with data cleaning, feature preprocessing, model training, evaluation, and an inference API.

## Overview

This repository provides a complete workflow for:

- loading raw delivery data
- cleaning and transforming the data
- splitting the dataset into train and test sets
- preprocessing features for machine learning
- training a regressor ensemble
- evaluating model performance
- registering and serving the trained model through a FastAPI application

The project uses DVC for pipeline orchestration, MLflow + Dagshub for experiment tracking and model registry integration, and a FastAPI service for online predictions.

## Features

- End-to-end data pipeline with DVC
- Data cleaning and feature engineering for delivery datasets
- Preprocessing pipeline with scaling, one-hot encoding, and ordinal encoding
- Stacking regression model with Random Forest and LightGBM
- MLflow experiment tracking and Dagshub integration
- FastAPI prediction API
- Docker support for deployment
- Automated sample prediction script

## Tech Stack

- Python 3.12
- FastAPI
- Uvicorn
- Pandas, NumPy
- Scikit-learn
- LightGBM
- MLflow
- Dagshub
- DVC
- Joblib
- Docker
- Pytest

## Project Architecture

The project follows a standard ML project layout with a reproducible pipeline:

1. Data ingestion
   - raw dataset is loaded from `data/raw/swiggy.csv`

2. Data cleaning
   - `src/data/data_cleaning.py`
   - cleans column names, handles invalid values, formats dates, computes pickup time and distance, and saves cleaned output

3. Data preparation
   - `src/data/data_preparation.py`
   - splits the cleaned dataset into train and test data

4. Feature preprocessing
   - `src/features/data_preprocessing.py`
   - applies scaling, encodings, and saves the fitted preprocessor

5. Model training
   - `src/models/train.py`
   - trains a `StackingRegressor` wrapped in `TransformedTargetRegressor`
   - saves trained model artifacts

6. Model evaluation and tracking
   - `src/models/evaluation.py`
   - evaluates on train/test data, logs metrics to MLflow, and writes `run_information.json`

7. Prediction API
   - `app.py`
   - loads the model from MLflow model registry and serves predictions through FastAPI

## Repository Structure

```text
.
├── app.py                      # FastAPI prediction service
├── Dockerfile                 # Docker container definition
├── dvc.yaml                   # DVC pipeline stages
├── LICENSE                    # License file
├── Makefile                   # Common commands
├── README.md                  # Project documentation
├── requirements.txt           # Base dependencies
├── requirements-dev.txt       # Development dependencies
├── requirements-dockers.txt   # Docker runtime dependencies
├── params.yaml                # Training hyperparameters
├── run_information.json       # Model metadata written by evaluation stage
├── scripts/
│   ├── data_clean_utils.py    # Helper cleaning functions used by the API
│   └── sample_predictions.py  # Example request script
├── src/
│   ├── data/
│   │   ├── data_cleaning.py
│   │   └── data_preparation.py
│   ├── features/
│   │   └── data_preprocessing.py
│   ├── models/
│   │   ├── evaluation.py
│   │   ├── register_model.py
│   │   └── train.py
│   └── visualization/
│       └── visualize.py
├── data/
│   ├── cleaned/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
│   ├── model.joblib
│   ├── preprocessor.joblib
│   ├── power_transformer.joblib
│   └── stacking_regressor.joblib
├── tests/
│   ├── test_model_perf.py
│   └── test_model_registry.py
├── docs/
├── notebooks/
├── reports/
└── setup.py
```

## Getting Started

### Prerequisites

- Python 3.12+
- pip
- virtualenv or conda (recommended)
- Git
- Docker (optional, for containerized deployment)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/vc-delivery-prediction.git
cd vc-delivery-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -U pip setuptools wheel
pip install -r requirements-dev.txt
pip install -e .
```

If you want the lightweight runtime dependencies only:

```bash
pip install -r requirements.txt
```

## Running the Data Pipeline

This project is organized as a DVC pipeline. The stages are defined in `dvc.yaml`.

### Run the pipeline manually

```bash
python src/data/data_cleaning.py
python src/data/data_preparation.py
python src/features/data_preprocessing.py
python src/models/train.py
python src/models/evaluation.py
```

### Run the entire DVC pipeline

```bash
dvc repro
```

## Running the Prediction API

Start the FastAPI service:

```bash
python app.py
```

Or use Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The service exposes:

- `GET /` → welcome endpoint
- `POST /predict` → prediction endpoint

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

A sample request body is generated from the raw data in `scripts/sample_predictions.py`.

## Model and Experiment Tracking

This project integrates with MLflow and Dagshub:

- training metrics are logged through `src/models/evaluation.py`
- model registry metadata is stored in `run_information.json`
- the inference app loads the production model from MLflow registry using `models:/<model-name>/Production`

## Docker

Build the image:

```bash
docker build -t vc-delivery-prediction .
```

Run the container:

```bash
docker run -p 8000:8000 vc-delivery-prediction
```

## Testing

Run the test suite:

```bash
pytest
```

The included tests validate the model performance threshold and registry usage.

## Sample Prediction Script

To run the sample inference script:

```bash
python scripts/sample_predictions.py
```

This script loads a sample record from the raw dataset, sends it to the local API, and prints the predicted delivery time.

## Notes

- The project currently includes a placeholder `LICENSE` file, so you should add an appropriate open-source license before publishing the repository publicly.
- `run_information.json` is generated during the evaluation stage and is used by the API to discover the registered model.
- Model artifacts are stored in the `models/` directory and can be reused for offline prediction workflows.

## Contributing

Contributions are welcome. If you would like to improve the project:

1. Create a feature branch
2. Make your changes
3. Run the relevant tests
4. Open a pull request with a clear summary

## License

This repository currently contains a `LICENSE` file, but the file does not include license text yet. Add the appropriate license before distributing the project publicly.
