import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(model: object, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    for name, value in metrics.items():
        logger.info("%s: %.4f", name, value)

    logger.info("\n%s", classification_report(y, y_pred, target_names=["Healthy", "Failure"]))
    return metrics
