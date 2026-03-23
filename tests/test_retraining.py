from __future__ import annotations

from core.retraining import select_best_trial


def test_select_best_trial_uses_lowest_val_rmse() -> None:
    trials = [
        {"ridge_alpha": 1e-4, "val_mean_rmse": 0.12},
        {"ridge_alpha": 1e-3, "val_mean_rmse": 0.10},
        {"ridge_alpha": 1e-2, "val_mean_rmse": 0.11},
    ]
    best = select_best_trial(trials)
    assert best["ridge_alpha"] == 1e-3
