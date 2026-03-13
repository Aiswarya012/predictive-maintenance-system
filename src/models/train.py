import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
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


def setup_mlflow(config: dict) -> None:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    logger.info("MLflow tracking URI: %s", config["mlflow"]["tracking_uri"])
    logger.info("MLflow experiment: %s", config["mlflow"]["experiment_name"])


def run_training_pipeline() -> None:
    config = load_config()
    seed = config["project"]["random_seed"]

    setup_mlflow(config)

    with mlflow.start_run(run_name=f"xgboost_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("dataset", "UCI AI4I 2020")

        df = preprocess(config)
        df = engineer_features(df)

        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("val_size", config["data"]["val_size"])
        mlflow.log_param("random_seed", seed)

        target_col = config["data"]["target_column"]
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, target_col, config["data"]["test_size"], config["data"]["val_size"], seed
        )

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size_actual", len(X_val))
        mlflow.log_param("test_size_actual", len(X_test))
        mlflow.log_param("features", list(X_train.columns))

        mlflow.log_params({
            f"model_{k}": v for k, v in config["model"]["params"].items()
        })

        scaler = fit_preprocessor(X_train)
        feature_names = list(X_train.columns)

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model = train_model(X_train_scaled, y_train, config["model"]["params"])

        logger.info("--- Validation Metrics ---")
        val_metrics = evaluate_model(model, X_val_scaled, y_val)
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)

        logger.info("--- Test Metrics ---")
        test_metrics = evaluate_model(model, X_test_scaled, y_test)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        save_dir = Path(config["model"]["save_dir"])
        model_path, preprocessor_path = save_artifacts(model, scaler, save_dir, feature_names)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(preprocessor_path))

        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        logger.info("MLflow run logged successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_training_pipeline()
