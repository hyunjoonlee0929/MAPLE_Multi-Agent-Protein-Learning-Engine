"""Property model retraining helpers."""

from __future__ import annotations


def select_best_trial(trials: list[dict]) -> dict:
    """Select best trial by validation mean RMSE (lower is better)."""
    if not trials:
        raise ValueError("trials must not be empty")
    return min(trials, key=lambda t: float(t.get("val_mean_rmse", 1e9)))
