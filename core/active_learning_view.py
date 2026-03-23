"""View helpers for active learning report visualization."""

from __future__ import annotations


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def active_learning_round_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload.get("rounds", []):
        fit = row.get("fit", {})
        val_mean = fit.get("val_metrics", {}).get("mean", {})
        train_mean = fit.get("train_metrics", {}).get("mean", {})
        acquired = row.get("acquired_batch", [])
        rows.append(
            {
                "round": int(row.get("round", 0)),
                "train_size": int(row.get("train_size", 0)),
                "val_size": int(row.get("val_size", 0)),
                "best_alpha": _safe_float(fit.get("best_alpha")),
                "val_rmse_mean": _safe_float(val_mean.get("rmse")),
                "val_mae_mean": _safe_float(val_mean.get("mae")),
                "val_r2_mean": _safe_float(val_mean.get("r2")),
                "train_rmse_mean": _safe_float(train_mean.get("rmse")),
                "acquired_count": len(acquired),
                "acq_mean": (
                    sum(_safe_float(item.get("acquisition")) for item in acquired) / len(acquired)
                    if acquired
                    else 0.0
                ),
                "pseudo_stability_mean": (
                    sum(_safe_float(item.get("pseudo_stability")) for item in acquired) / len(acquired)
                    if acquired
                    else 0.0
                ),
                "pseudo_activity_mean": (
                    sum(_safe_float(item.get("pseudo_activity")) for item in acquired) / len(acquired)
                    if acquired
                    else 0.0
                ),
            }
        )
    return rows


def active_learning_acquisition_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload.get("rounds", []):
        round_idx = int(row.get("round", 0))
        for item in row.get("acquired_batch", []):
            rows.append(
                {
                    "round": round_idx,
                    "sequence": item.get("sequence"),
                    "acquisition": _safe_float(item.get("acquisition")),
                    "pred_mean": _safe_float(item.get("pred_mean")),
                    "novelty": _safe_float(item.get("novelty")),
                    "pseudo_stability": _safe_float(item.get("pseudo_stability")),
                    "pseudo_activity": _safe_float(item.get("pseudo_activity")),
                }
            )
    return rows
