"""Evaluation helpers shared by the classifier notebook."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)
