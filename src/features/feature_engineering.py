# feature engineering
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging
import pickle
from pathlib import Path

# logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("feature_engineering_errors.log")
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


# Load parameters from a yaml file
def load_params(file_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file with error logging."""
    try:
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
        logger.info(f"YAML configuration loaded successfully from {file_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"YAML file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error : {e}")
        raise


# Save the DataFrame to a CSV file with logging and error handling.
def data_save(data: pd.DataFrame, save_path: Path) -> None:
    try:
        if data.empty:
            raise ValueError("Cannot save an emtpy DataFrame.")
        data.to_csv(save_path, index=False)
        logger.info(f"Data saved successfully to {save_path }")
    except FileNotFoundError as e:
        logger.error(f"File path not found: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied while saving file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving object to {save_path,e}")
        raise


def apply_bow(
    train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int
) -> tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data["content"].fillna("").values
        y_train = train_data["sentiment"].values
        X_test = test_data["content"].fillna("").values
        y_test = test_data["sentiment"].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        # Save the vectorizer
        pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

        logger.info("Bag of Words applied and data transformed successfully.")
        return train_df, test_df

    except Exception as e:
        logger.error(f"Error during Bag of Words transformation: {e}")
        raise


def main():
    try:
        root_path = Path(__file__).parent.parent.parent

        # params yaml file read
        params = load_params(file_path="params.yaml")
        max_features = params['feature_engineering']['max_features']

        # data load
        train_data = data_load(data_path=root_path/'data' / 'interim' / 'train_processed.csv')
        test_data = data_load(data_path=root_path/'data' / 'interim' / 'test_processed.csv')

        # apply bow funciton
        train_df, test_df = apply_bow(train_data=train_data, test_data=test_data, max_features=max_features)

        file_path = root_path / 'data' / 'processed'
        file_path.mkdir(parents=True, exist_ok=True)
        # save data
        data_save(data=train_df, save_path=file_path / 'train_bow.csv')
        data_save(data=test_df, save_path=file_path / 'test_bow.csv')

    except Exception as e:
        logger.error(f"Failed to complete the feature engineering process: {e}")
        raise


if __name__ == "__main__":
    main()
