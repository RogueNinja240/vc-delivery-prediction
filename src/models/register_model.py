import mlflow
import dagshub
import json 
from pathlib import Path
from mlflow import MlflowClient
import logging

# Set up logging
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='RogueNinja240', repo_name='vc-delivery-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/vc-delivery-prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

if __name__ =="__main__":
    root_path = Path(__file__).parent.parent.parent 
    run_info_path = root_path / "run_information.json"
    
    # 1. Load info (We do this to keep the DVC dependency chain intact)
    run_info = load_model_information(run_info_path)
    model_name = run_info["model_name"]

    # 2. Skip re-registering! Just find the latest version.
    client = MlflowClient()
    
    # Fetch all versions of this specific model
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    # Find the highest version number (evaluate.py just created version 7!)
    latest_version = max([int(v.version) for v in model_versions])

    logger.info(f"The latest model version in model registry is {latest_version} and name is {model_name}")
    
    # 3. Promote to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=str(latest_version),
        stage='Staging'
    )
    logger.info("Model Pushed to staging state")