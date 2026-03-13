FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    xgboost==2.1.3 \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    pydantic==2.10.4 \
    pyyaml==6.0.2 \
    joblib==1.4.2

COPY configs/ configs/
COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
