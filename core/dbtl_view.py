"""View helpers for DBTL ingestion report visualization."""

from __future__ import annotations


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def dbtl_summary_row(payload: dict) -> dict:
    stats = payload.get("merge_stats", {})
    fit = payload.get("fit", {}) or {}
    val_mean = fit.get("val_metrics", {}).get("mean", {})
    train_mean = fit.get("train_metrics", {}).get("mean", {})
    return {
        "retrain_triggered": bool(payload.get("retrain_triggered", False)),
        "imported_records": int(stats.get("imported_records", 0)),
        "train_added": int(stats.get("train_added", 0)),
        "train_updated": int(stats.get("train_updated", 0)),
        "val_added": int(stats.get("val_added", 0)),
        "val_updated": int(stats.get("val_updated", 0)),
        "train_size": int(payload.get("train_size", 0)),
        "val_size": int(payload.get("val_size", 0)),
        "val_rmse_mean": _safe_float(val_mean.get("rmse")),
        "train_rmse_mean": _safe_float(train_mean.get("rmse")),
        "checkpoint": payload.get("checkpoint"),
    }


def dbtl_trial_rows(payload: dict) -> list[dict]:
    fit = payload.get("fit", {}) or {}
    rows = []
    for item in fit.get("trials", []):
        rows.append(
            {
                "ridge_alpha": _safe_float(item.get("ridge_alpha")),
                "val_mean_rmse": _safe_float(item.get("val_mean_rmse")),
                "val_mean_mae": _safe_float(item.get("val_mean_mae")),
                "val_mean_r2": _safe_float(item.get("val_mean_r2")),
                "val_mean_pearson": _safe_float(item.get("val_mean_pearson")),
            }
        )
    return rows
