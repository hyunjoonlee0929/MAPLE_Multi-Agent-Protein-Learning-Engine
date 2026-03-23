from __future__ import annotations

import numpy as np

from utils.metrics import evaluate_property_metrics


def test_evaluate_property_metrics_returns_expected_keys() -> None:
    y_true = np.array(
        [
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )
    y_pred = np.array(
        [
            [0.52, 0.58],
            [0.68, 0.83],
            [0.91, 0.98],
        ],
        dtype=np.float32,
    )

    metrics = evaluate_property_metrics(y_true, y_pred)
    assert "stability" in metrics
    assert "activity" in metrics
    assert "mean" in metrics
    assert "rmse" in metrics["mean"]
    assert "mae" in metrics["mean"]
    assert "r2" in metrics["mean"]
    assert "pearson" in metrics["mean"]
