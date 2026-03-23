import pytest 
import mlflow 
from mlflow import MlflowClient
import dagshub 
import json 

dagshub.init(repo_owner='RogueNinja240', repo_name='vc-delivery-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/vc-delivery-prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

model_name = load_model_information("run_information.json")["model_name"]

@pytest.mark.parametrize(argnames="model_name,stage",argvalues=[(model_name,"Staging")])
def test_load_model_from_registry(model_name,stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name,stages=[stage])
    latest_version = latest_versions[0].version if latest_versions else None 
    assert latest_version is not None,f"No model at {stage} stage"

    model_path = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_path)
    assert model is not None, f"Model failed to load from registry"
    print(f"The {model_name} with version {latest_version} was loaded successfully")
    