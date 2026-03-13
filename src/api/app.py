import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionResponse, SensorInput

logger = logging.getLogger(__name__)

model_artifacts: dict[str, Any] = {}


def load_config(config_path: Path = Path("configs/config.yaml")) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_input(sensor: SensorInput, feature_names: list[str]) -> pd.DataFrame:
    data = {
        "Air temperature [K]": sensor.air_temperature_k,
        "Process temperature [K]": sensor.process_temperature_k,
        "Rotational speed [rpm]": sensor.rotational_speed_rpm,
        "Torque [Nm]": sensor.torque_nm,
        "Tool wear [min]": sensor.tool_wear_min,
    }

    data["temp_diff"] = sensor.process_temperature_k - sensor.air_temperature_k
    data["power"] = sensor.torque_nm * sensor.rotational_speed_rpm * (2 * 3.14159 / 60)
    data["wear_torque_interaction"] = sensor.tool_wear_min * sensor.torque_nm

    data["Type_L"] = 1 if sensor.machine_type == "L" else 0
    data["Type_M"] = 1 if sensor.machine_type == "M" else 0

    df = pd.DataFrame([data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    model_path = Path(config["api"]["model_path"])
    preprocessor_path = Path(config["api"]["preprocessor_path"])

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Train the model first. "
            f"Expected: {model_path}, {preprocessor_path}"
        )

    model_artifact = joblib.load(model_path)
    scaler = joblib.load(preprocessor_path)

    model_artifacts["model"] = model_artifact["model"]
    model_artifacts["feature_names"] = model_artifact["feature_names"]
    model_artifacts["scaler"] = scaler

    logger.info("Model loaded from %s", model_path)
    yield
    model_artifacts.clear()


app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(sensor: SensorInput) -> PredictionResponse:
    try:
        df = prepare_input(sensor, model_artifacts["feature_names"])
        scaled = model_artifacts["scaler"].transform(df)
        prediction = model_artifacts["model"].predict(scaled)[0]
        probability = model_artifacts["model"].predict_proba(scaled)[0][1]

        return PredictionResponse(
            prediction="Failure" if prediction == 1 else "Healthy",
            failure_probability=round(float(probability), 4),
        )
    except KeyError as e:
        raise HTTPException(status_code=503, detail="Model not loaded") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
