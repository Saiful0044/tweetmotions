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
def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )

        logger.debug(
            f"Model {model_name} version {model_version.version} registered and transitioned to Staging."
        )
    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise


# -------------------- Main Function --------------------
def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "MyRegisterModel"
        register_model(model_name=model_name, model_info=model_info)

    except Exception as e:
        logger.error(f"Failed to complete the model registration process: {str(e)}")


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
