import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

load_dotenv()
# Set credentials
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

# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# data load
def data_load(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logger.debug(f"Data loaded successfully from {data_path}")
        return data

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {data_path}: {e}")
        raise

    except FileNotFoundError as e:
        logger.error(f"File not found at {data_path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading data from {data_path}: {e}")
        raise


def load_model(model_path: str):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as fe:
        logger.error("FileNotFoundError: %s", fe)
        raise

    except Exception as e:
        logger.exception(
            f"Error occurred while loading the model from {model_path}: {e}"
        )
        raise


def evaluation_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        logger.info("Starting model evaluation...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving the metrics: {e}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as f:
            json.dump(model_info, f, indent=4)
        logger.debug(f"Model info saved to {file_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving the model info: {e}")
        raise


def main():
    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            root_path = Path(__file__).parent.parent.parent
            model = load_model(root_path / "models" / "model.pkl")
            test_data = data_load(root_path / "data" / "processed" / "test_bow.csv")

            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            metrics = evaluation_model(model=model, X_test=X_test, y_test=y_test)

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters to MLflow
            if hasattr(model, "get_params"):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            save_metrics(metrics, "reports/metrics.json")
            # save model info
            save_model_info(run.info.run_id, "model", "reports/experiment_info.json")

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")

            # Log the metrics file to MLflow
            mlflow.log_artifact("reports/metrics.json")

            # Log the model info file to MLflow
            mlflow.log_artifact("reports/experiment_info.json")

            # Log the evaluation errors log file to MLflow
            mlflow.log_artifact("model_evaluation_errors.log")

        except Exception as e:
            logger.error(f"Failed to complete the model evaluation process: {e}")
            raise


if __name__ == "__main__":
    main()
