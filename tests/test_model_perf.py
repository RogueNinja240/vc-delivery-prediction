import pytest 
import mlflow 
import dagshub
import json 
from pathlib import Path 
from sklearn.pipeline import Pipeline 
import joblib 
import pandas as pd 
from sklearn.metrics import mean_absolute_error

dagshub.init(repo_owner='RogueNinja240', repo_name='vc-delivery-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/vc-delivery-prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer # Fixed: Changed 'raise' to 'return'

model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

# Fixed: Removed one .parent to stay inside the project root
root_path = Path(__file__).parent.parent 
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

model_pipe = Pipeline(steps=[('preprocess',preprocessor),('regressor',model)])

# Because root_path is fixed, this will now correctly point to vc-delivery-prediction/data/interim/test.csv
test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe,test_data_path,threshold_error",argvalues=[(model_pipe,test_data_path,5)])
def test_model_performance(model_pipe,test_data_path,threshold_error):
    df = pd.read_csv(test_data_path)
    df.dropna(inplace=True)
    X = df.drop(columns=['time_taken'])
    y = df['time_taken']
    y_pred = model_pipe.predict(X)
    mean_error = mean_absolute_error(y,y_pred)
    
    assert mean_error <= threshold_error, "The model doesn't pass the performance threshold value"
    
    print("the avg error is", mean_error)
    print(f"The {model_name} model passed the performance test")