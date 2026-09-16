import json
from pathlib import Path
import joblib
import mlflow
import mlflow.client
from mlflow import MlflowClient
import pandas as pd
from pydantic import BaseModel, Field
from sklearn import set_config
from sklearn.pipeline import Pipeline
import uvicorn
from fastapi import FastAPI, HTTPException

import dagshub
from scripts.data_clean_utils import perform_data_cleaning

# Set output to pandas
set_config(transform_output="pandas")

# Initialize DagsHub and MLflow
dagshub.init(repo_owner="RogueNinja240", repo_name="vc-delivery-prediction", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/vc-delivery-prediction.mlflow")


class Data(BaseModel):
    ID: str = Field(..., json_schema_extra={"example": "0x4607"})
    Delivery_person_ID: str = Field(..., json_schema_extra={"example": "INDORES13DEL02"})
    Delivery_person_Age: str = Field(..., json_schema_extra={"example": "37"})
    Delivery_person_Ratings: str = Field(..., json_schema_extra={"example": "4.9"})
    Restaurant_latitude: float = Field(..., json_schema_extra={"example": 22.745049})
    Restaurant_longitude: float = Field(..., json_schema_extra={"example": 75.892471})
    Delivery_location_latitude: float = Field(..., json_schema_extra={"example": 22.765049})
    Delivery_location_longitude: float = Field(..., json_schema_extra={"example": 75.912471})
    Order_Date: str = Field(..., json_schema_extra={"example": "19-03-2022"})
    Time_Orderd: str = Field(..., json_schema_extra={"example": "11:30:00"})
    Time_Order_picked: str = Field(..., json_schema_extra={"example": "11:45:00"})
    Weatherconditions: str = Field(..., json_schema_extra={"example": "conditions Sunny"})
    Road_traffic_density: str = Field(..., json_schema_extra={"example": "High"})
    Vehicle_condition: int = Field(..., json_schema_extra={"example": 2})
    Type_of_order: str = Field(..., json_schema_extra={"example": "Snack"})
    Type_of_vehicle: str = Field(..., json_schema_extra={"example": "motorcycle"})
    multiple_deliveries: str = Field(..., json_schema_extra={"example": "0"})
    Festival: str = Field(..., json_schema_extra={"example": "No"})
    City: str = Field(..., json_schema_extra={"example": "Metropolitian"})


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


# Load model info and setup MLflow artifacts
model_name = load_model_information("run_information.json")["model_name"]
stage = "Production"
model_path = f"models:/{model_name}/{stage}"

model = mlflow.sklearn.load_model(model_path)
preprocessor = load_transformer("models/preprocessor.joblib")

model_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", model)
])

# Load sample dataset into memory for live simulation
SAMPLE_CSV_PATH = Path("data/sample_raw_test.csv")
sample_data = None
if SAMPLE_CSV_PATH.exists():
    sample_data = pd.read_csv(SAMPLE_CSV_PATH)

app = FastAPI(
    title="Swiggy Food Delivery ETA Prediction",
    description="End-to-End MLOps Pipeline with Stacking Regressor & Feature Preprocessing",
    version="1.0.0"
)


@app.get(path="/")
def home():
    return {
        "message": "Welcome to the Swiggy Food Delivery Time Prediction App",
        "endpoints": {
            "demo": "/predict/demo (GET: Simulates a random delivery)",
            "predict": "/predict (POST: Custom input payload)",
            "docs": "/docs (Interactive Swagger UI)"
        }
    }


@app.get(path="/predict/demo", tags=["Demo"])
def demo_prediction():
    """
    Simulates a live delivery order by sampling 1 row from the test set,
    running it through preprocessing, and returning the estimated delivery time
    alongside the actual historical time for accuracy comparison.
    """
    if sample_data is None or sample_data.empty:
        raise HTTPException(status_code=500, detail="Sample dataset not found.")

    cleaned_data = pd.DataFrame()
    actual_eta = None
    random_row = None
    
    # Keep sampling a new row until we find one that survives the data cleaning process
    while cleaned_data.empty:
        random_row = sample_data.sample(n=1)
        
        target_col = "Time_taken(min)"
        if target_col in random_row.columns:
            raw_time_string = str(random_row[target_col].values[0])
            # Strip out the "(min)" text and any extra spaces
            actual_eta = float(raw_time_string.replace("(min)", "").strip())
            
            # Drop the target column so the model only gets the input features
            random_row = random_row.drop(columns=[target_col])
        
        # If the row has NaNs, this will return an empty dataframe, triggering the loop to try again
        cleaned_data = perform_data_cleaning(random_row.copy())

    predicted_eta = float(model_pipe.predict(cleaned_data)[0])

    return {
        "status": "success",
        "predicted_eta_minutes": round(predicted_eta, 2),
        "actual_eta_minutes": round(actual_eta, 2) if actual_eta else None,
        "error_margin_minutes": round(abs(predicted_eta - actual_eta), 2) if actual_eta else None,
        "simulated_order_features": random_row.to_dict(orient="records")[0]
    }

@app.post(path="/predict", tags=["Inference"])
def do_predictions(data: Data):
    pred_data = pd.DataFrame([data.model_dump()])
    cleaned_data = perform_data_cleaning(pred_data)
    predictions = float(model_pipe.predict(cleaned_data)[0])
    return {
        "status": "success",
        "predicted_eta_minutes": round(predictions, 2)
    }


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)