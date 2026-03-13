# Predictive Maintenance System

An end-to-end MLOps project that predicts industrial machine failures using sensor data. Built with XGBoost, FastAPI, Streamlit, and Docker.

---

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌────────────────┐
│  UCI AI4I   │───▶│ Preprocessing│───▶│   Training   │───▶│  Model   │───▶│  FastAPI /      │
│  Dataset    │    │ + Features   │    │  (XGBoost)   │    │ Registry │    │  Streamlit UI  │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────┘    └────────────────┘
                                             │                                    │
                                       ┌─────▼─────┐                      ┌──────▼──────┐
                                       │ Evaluation │                      │   Docker    │
                                       │  Metrics   │                      │  Container  │
                                       └───────────┘                      └─────────────┘
```

---

## Dataset

**UCI AI4I 2020 Predictive Maintenance Dataset** — 10,000 synthetic industrial sensor records.

| Feature | Description | Unit |
|---|---|---|
| Air temperature | Ambient air temperature | Kelvin |
| Process temperature | Process operating temperature | Kelvin |
| Rotational speed | Machine RPM | rpm |
| Torque | Torque applied to machine | Nm |
| Tool wear | Cumulative tool usage time | minutes |
| Type | Machine quality variant | L / M / H |

| Engineered Feature | Formula |
|---|---|
| temp_diff | Process temperature − Air temperature |
| power | Torque × RPM × (2π / 60) |
| wear_torque_interaction | Tool wear × Torque |

**Target:** `Machine failure` — binary classification (0 = Healthy, 1 = Failure)

---

## Project Structure

```
predictive-maintenance-system/
├── configs/
│   └── config.yaml                # All configuration (no hardcoded values)
├── data/
│   ├── raw/                       # Raw dataset (gitignored)
│   └── processed/                 # Preprocessed output
├── models/                        # Saved model artifacts (gitignored)
├── src/
│   ├── data/
│   │   └── preprocess.py          # Data loading, cleaning, encoding
│   ├── features/
│   │   └── engineering.py         # Feature engineering
│   ├── models/
│   │   └── train.py               # Training pipeline
│   ├── evaluation/
│   │   └── evaluate.py            # Multi-metric evaluation
│   ├── api/
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── app.py                 # FastAPI prediction service
│   └── app/
│       └── streamlit_app.py       # Streamlit dashboard
├── pipelines/
│   └── train_pipeline.py          # Training entry point
├── scripts/
│   └── download_data.py           # Dataset downloader
├── tests/
│   └── test_api.py                # API tests
├── Dockerfile
├── pyproject.toml
└── .gitignore
```

---

## Tech Stack

| Component | Tool |
|---|---|
| ML Model | XGBoost |
| Preprocessing | Scikit-learn (StandardScaler) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data Validation | Pydantic |
| Configuration | YAML |
| Containerization | Docker |
| Language | Python 3.11 |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/Aiswarya012/predictive-maintenance-system.git
cd predictive-maintenance-system
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Download Dataset

```bash
python scripts/download_data.py
```

### 5. Train the Model

```bash
python -m pipelines.train_pipeline
```

Training output includes evaluation metrics on both validation and test sets:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Full classification report

Model artifacts are saved with timestamps for versioning:
```
models/
├── model_20260313_042209.joblib
├── model_latest.joblib
├── preprocessor_20260313_042209.joblib
└── preprocessor_latest.joblib
```

### 6. Run the FastAPI Server

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (interactive) |
| POST | `/predict` | Predict machine failure |

### 7. Run the Streamlit Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Opens at `http://localhost:8501` with interactive sliders for all sensor inputs.

---

## API Usage

### Healthy Machine Example

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
    "machine_type": "M"
  }'
```

**Response:**

```json
{
    "prediction": "Healthy",
    "failure_probability": 0.0002,
    "status": "success"
}
```

### Machine Failure Example

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 300.0,
    "process_temperature_k": 312.0,
    "rotational_speed_rpm": 1300,
    "torque_nm": 75.0,
    "tool_wear_min": 240,
    "machine_type": "L"
  }'
```

**Response:**

```json
{
    "prediction": "Failure",
    "failure_probability": 0.9999,
    "status": "success"
}
```

### Request Schema

| Field | Type | Description |
|---|---|---|
| `air_temperature_k` | float | Air temperature in Kelvin (typical: 295–305) |
| `process_temperature_k` | float | Process temperature in Kelvin (typical: 305–315) |
| `rotational_speed_rpm` | float | Rotational speed (typical: 1000–3000) |
| `torque_nm` | float | Torque in Nm (typical: 3–80) |
| `tool_wear_min` | float | Tool wear in minutes (typical: 0–250) |
| `machine_type` | string | Machine quality type: `L`, `M`, or `H` |

### Response Schema

| Field | Type | Description |
|---|---|---|
| `prediction` | string | `Healthy` or `Failure` |
| `failure_probability` | float | Probability of failure (0.0 – 1.0) |
| `status` | string | `success` |

---

## Docker

### Build and Run Locally

```bash
docker build -t predictive-maintenance:latest .
docker run -d -p 8000:8000 --name pred-maintenance predictive-maintenance:latest
```

### Verify

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature_k":298.1,"process_temperature_k":308.6,"rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":0,"machine_type":"M"}'
```

### Stop

```bash
docker stop pred-maintenance && docker rm pred-maintenance
```

---

## Deploy on Linux VM

```bash
# 1. SSH into VM
ssh user@your-vm-ip

# 2. Install Docker
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect

# 3. Clone the project
git clone https://github.com/Aiswarya012/predictive-maintenance-system.git
cd predictive-maintenance-system

# 4. Train the model on the VM
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/download_data.py
python -m pipelines.train_pipeline
deactivate

# 5. Build and run Docker container
docker build -t predictive-maintenance:latest .
docker run -d -p 8000:8000 --restart unless-stopped --name pred-maintenance predictive-maintenance:latest

# 6. Verify
curl http://localhost:8000/health
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Configuration

All configuration is centralized in `configs/config.yaml`:

- **Data paths** — raw/processed file locations
- **Feature lists** — categorical and numerical columns
- **Model hyperparameters** — XGBoost settings
- **API settings** — host, port, model paths

No hardcoded values in source code.
