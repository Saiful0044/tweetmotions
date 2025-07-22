import numpy as np
import pandas as pd
import json
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Saiful0044"
repo_name = "tweetmotions"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# -------------------- Logging Configuration --------------------
logger = logging.getLogger("model registry")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registry_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------- Load Model Info --------------------
def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.info(f"Model info loaded from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error occurred while loading the model info: {str(e)}"
        )
        raise


# -------------------- Register Model --------------------
def register_model(model_name: str):
    """Register the latest valid MLflow model to the Model Registry."""
    try:
        # mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = MlflowClient()
        experiment = client.get_experiment_by_name("dvc-pipeline")

        # Get latest run
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                order_by= ["start_time DESC"],
                                max_results=1)
        run_id = runs[0].info.run_id
        logger.info(f"Latest run id is found: {run_id}")
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model", name=model_name
        )

        client.set_registered_model_alias(
            name=model_name,
            version=model_version.version,
            alias="production",
            
        )
        logger.info(
            f"Model {model_name} version {model_version.version} transitioned to 'Production'"
        )
    except Exception as e:
        logger.error(f"Error during model registration: {str(e)}")
        raise


# -------------------- Main Function --------------------
def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "MyRegisterModel"
        register_model(model_name=model_name)

    except Exception as e:
        logger.error(f"Failed to complete the model registration process: {str(e)}")


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
