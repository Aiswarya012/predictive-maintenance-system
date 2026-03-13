from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


VALID_PAYLOAD = {
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
    "machine_type": "M",
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.skipif(
    not Path("models/model_latest.joblib").exists(),
    reason="Model not trained yet",
)
def test_predict(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Healthy", "Failure"]
    assert 0 <= data["failure_probability"] <= 1
    assert data["status"] == "success"


@pytest.mark.skipif(
    not Path("models/model_latest.joblib").exists(),
    reason="Model not trained yet",
)
def test_predict_failure_scenario(client):
    payload = {
        "air_temperature_k": 305.0,
        "process_temperature_k": 315.0,
        "rotational_speed_rpm": 1200,
        "torque_nm": 70.0,
        "tool_wear_min": 220,
        "machine_type": "L",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
