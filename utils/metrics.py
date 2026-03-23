"""Regression metrics for MAPLE property model evaluation."""

from __future__ import annotations

import numpy as np


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    return float(np.sqrt(np.mean(err**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    return _safe_mean(np.abs(err))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = _safe_mean(y_true)
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    if std_true <= 1e-12 or std_pred <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_property_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-target and averaged regression metrics.

    Input shape: [N, 2], target order [stability, activity]
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError("y_true must have shape [N, 2]")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred shape must match y_true")

    out = {}
    names = ["stability", "activity"]
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "r2": r2(yt, yp),
            "pearson": pearson_corr(yt, yp),
        }

    out["mean"] = {
        "rmse": float(np.mean([out["stability"]["rmse"], out["activity"]["rmse"]])),
        "mae": float(np.mean([out["stability"]["mae"], out["activity"]["mae"]])),
        "r2": float(np.mean([out["stability"]["r2"], out["activity"]["r2"]])),
        "pearson": float(np.mean([out["stability"]["pearson"], out["activity"]["pearson"]])),
    }
    return out
