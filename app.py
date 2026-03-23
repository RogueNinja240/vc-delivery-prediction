from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd 
import mlflow
import json 
import joblib 
from mlflow import MlflowClient
from sklearn import set_config
from pathlib import Path
from scripts.data_clean_utils import perform_data_cleaning
# Set output to pandas
set_config(transform_output='pandas')

import dagshub
import mlflow.client

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='RogueNinja240', repo_name='vc-delivery-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/vc-delivery-prediction.mlflow")

class Data(BaseModel):
    ID:str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float 
    Restaurant_longitude: float 
    Delivery_location_latitude: float 
    Delivery_location_longitude: float 
    Order_Date: str
    Time_Ordered: str 
    Time_Order_picked: str 
    Weatherconditions: str
    Road_traffic_density: str 
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str 
    multiple_deliveries: str
    Festival:str
    City:str

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model 

# Load model info and setup MLflow client
model_name = load_model_information("run_information.json")['model_name']
stage = "Staging"

# MLflow automatically fetches the latest version in the specified stage!
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

# Load local preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_model(preprocessor_path)

# Combine into an inference pipeline
model_pipe = Pipeline(steps=[('preprocess', preprocessor), ('regressor', model)])

app = FastAPI()

@app.get(path="/")
def home():
    return {"message": "Food Delivery Time Prediction API is live!"}

@app.post(path='/predict')
def do_predictions(data: Data):
    # Convert incoming Pydantic data to a DataFrame for the pipeline
    pred_data = pd.DataFrame({
    'Delivery_person_Age': data.Delivery_person_Age,
    'Delivery_person_Ratings': data.Delivery_person_Ratings,
    'Restaurant_latitude': data.Restaurant_latitude,
    'Restaurant_longitude': data.Restaurant_longitude,
    'Delivery_location_latitude': data.Delivery_location_latitude,
    'Delivery_location_longitude': data.Delivery_location_longitude,
    'Order_Date': data.Order_Date,
    'Time_Ordered': data.Time_Ordered,
    'Time_Order_picked': data.Time_Order_picked,
    'Weatherconditions': data.Weatherconditions,
    'Road_traffic_density': data.Road_traffic_density,
    'Vehicle_condition': data.Vehicle_condition,
    'Type_of_order': data.Type_of_order,
    'Type_of_vehicle': data.Type_of_vehicle,
    'multiple_deliveries': data.multiple_deliveries,
    'Festival': data.Festival,
    'City': data.City
}, index=[0])

    cleaned_data = perform_data_cleaning(pred_data)
    # Run the prediction
    predictions = model_pipe.predict(pred_data)[0]
    
    # Return as a properly formatted JSON dictionary
    return {"predicted_delivery_time": float(predictions)}

if __name__ =="__main__":
    uvicorn.run(app="app:app", host="127.0.0.1", port=8000)