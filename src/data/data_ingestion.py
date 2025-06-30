import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

# loggging configuration
# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.DEBUG)

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


# data preprocess
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        df = data.copy()
        if 'tweet_id' in df.columns:
            df = df.drop(columns=['tweet_id'])

        # filter for only happiness and sadness
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        # Encode sentiments
        sent_mapping = {'happiness': 1, 'sadness': 0}
        df['sentiment'] = df['sentiment'].map(sent_mapping)

        logger.info("Data preprocessing completed successfully")

        return df
    except Exception:
        logger.error("Unexpected error during data preprocessing.")
        raise


# Load parameters from a yaml file
def load_params(file_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file with error logging."""
    try:
        with open(file_path, 'r') as f:
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


def main():
    root_path = Path(__file__).parent.parent.parent
    # load data
    data = data_load(data_path="https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
    # data preprocess
    data = preprocess_data(data=data)
    # loam yaml
    params = load_params(file_path='params.yaml')
    train, test = train_test_split(
        data, test_size=params["data_ingestion"]["test_size"], random_state=42
    )
    # save data
    file_path = root_path / "data" / 'raw'
    file_path.mkdir(exist_ok=True, parents=True)
    data_save(data=train, save_path=file_path / 'train.csv')
    data_save(data=test, save_path=file_path / 'test.csv')


if __name__ == '__main__':
    main()