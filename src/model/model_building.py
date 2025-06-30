import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging
from pathlib import Path

# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_building_errors.log")
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


def trian_model(X_trian: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    try:
        logger.info("Building the model...")
        clf = LogisticRegression(C=1, solver= 'liblinear', penalty= 'l2')
        clf.fit(X_trian, y_train)
        logger.info("Model training successfully.")
        return clf

    except Exception as e:
        logger.exception(f"Error occurred while building the model: {e}")
        raise


# save_model
def save_model(model, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved successfully to {file_path}")
    
    except Exception as e:
        logger.exception(f"Failed to save model to {file_path}: {e}")
        raise


def main():
    try:
        root_path = Path(__file__).parent.parent.parent
        train_data = data_load(root_path / 'data' / 'processed' / 'train_bow.csv')
        X_trian = train_data.iloc[:, :-1].values
        y_trian = train_data.iloc[:, -1].values

        clf = trian_model(X_trian=X_trian, y_train=y_trian)

        save_model(clf, root_path / 'models' / 'model.pkl')

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        raise


if __name__ == "__main__":
    main()