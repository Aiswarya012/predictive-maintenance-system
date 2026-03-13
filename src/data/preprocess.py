import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("configs/config.yaml")) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(path: Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def drop_unnecessary_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    logger.info("Dropping columns: %s", existing)
    return df.drop(columns=existing)


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=np.int32)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning("Found %d missing values, filling with median", missing)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def preprocess(config: dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    df = load_raw_data(raw_path)
    df = drop_unnecessary_columns(df, config["data"]["drop_columns"])
    df = handle_missing_values(df)
    df = encode_categorical(df, config["features"]["categorical_columns"])

    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("Saved processed data to %s", processed_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cfg = load_config()
    preprocess(cfg)
