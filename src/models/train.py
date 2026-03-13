import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.preprocess import load_config, preprocess
from src.evaluation.evaluate import evaluate_model
from src.features.engineering import engineer_features

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=seed, stratify=y_train_val
    )

    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocessor(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def train_model(X_train: np.ndarray, y_train: pd.Series, params: dict) -> XGBClassifier:
    logger.info("Training XGBoost with params: %s", params)
    model = XGBClassifier(**params, use_label_encoder=False)
    model.fit(X_train, y_train)
    return model


def save_artifacts(
    model: XGBClassifier, scaler: StandardScaler, save_dir: Path, feature_names: list[str]
) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    model_path = save_dir / f"model_{timestamp}.joblib"
    preprocessor_path = save_dir / f"preprocessor_{timestamp}.joblib"
    latest_model_path = save_dir / "model_latest.joblib"
    latest_preprocessor_path = save_dir / "preprocessor_latest.joblib"

    model_artifact = {"model": model, "feature_names": feature_names}

    joblib.dump(model_artifact, model_path)
    joblib.dump(scaler, preprocessor_path)
    joblib.dump(model_artifact, latest_model_path)
    joblib.dump(scaler, latest_preprocessor_path)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved preprocessor to %s", preprocessor_path)
    return model_path, preprocessor_path


def run_training_pipeline() -> None:
    config = load_config()
    seed = config["project"]["random_seed"]

    df = preprocess(config)
    df = engineer_features(df)

    target_col = config["data"]["target_column"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_col, config["data"]["test_size"], config["data"]["val_size"], seed
    )

    scaler = fit_preprocessor(X_train)
    feature_names = list(X_train.columns)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = train_model(X_train_scaled, y_train, config["model"]["params"])

    logger.info("--- Validation Metrics ---")
    evaluate_model(model, X_val_scaled, y_val)

    logger.info("--- Test Metrics ---")
    evaluate_model(model, X_test_scaled, y_test)

    save_dir = Path(config["model"]["save_dir"])
    save_artifacts(model, scaler, save_dir, feature_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_training_pipeline()
