import numpy as np
import pandas as pd
import logging
import os
import re
import nltk
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("stopwords")

# logging configuration
logger = logging.getLogger("data_transformation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("transformation_errors.log")
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


# difine text preprocessing functions
def lemmatization(text):
    """Lemmatize the text"""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """remove stop words from the text"""
    stop_words = set(stopwords.words("english"))
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """remove numbers from the text"""
    text = "".join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    """Convert text to lower case"""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """remove punctuations from the text"""
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace(":", "")
    text = re.sub("\s+", " ", text).strip()
    return text


def removes_urls(text):
    """Remove urls from the text"""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def normalize_text(df):
    """Normalize the text data"""
    try:
        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(remove_stop_words)
        df["content"] = df["content"].apply(removing_numbers)
        df["content"] = df["content"].apply(removing_punctuations)
        df["content"] = df["content"].apply(removes_urls)
        df["content"] = df["content"].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
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
    try:
        root_path = Path(__file__).parent.parent.parent
        # fetch the data from data/raw
        train_data = data_load(data_path=root_path / 'data' / 'raw' / 'train.csv')
        test_data = data_load(data_path=root_path / "data" / "raw" / "test.csv")
        logger.info('Data loaded properly')
        # transform the data
        train_procesed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # store the data inside data/interim
        data_path = root_path / 'data' / 'interim'
        data_path.mkdir(parents=True, exist_ok=True)
        data_save(data=train_procesed_data, save_path=data_path / "train_processed.csv")
        data_save(data=test_processed_data, save_path=data_path / 'test_processed.csv')

    except Exception:
        logger.error("Failed to complete the data transformation process:")
        raise


if __name__ == '__main__':
    main()
