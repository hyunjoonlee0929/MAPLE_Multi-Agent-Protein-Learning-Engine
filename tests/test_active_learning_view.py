from __future__ import annotations

from core.active_learning_view import active_learning_acquisition_rows, active_learning_round_rows


def test_active_learning_round_rows_parses_metrics() -> None:
    payload = {
        "rounds": [
            {
                "round": 0,
                "train_size": 20,
                "val_size": 4,
                "fit": {
                    "best_alpha": 0.001,
                    "train_metrics": {"mean": {"rmse": 0.01}},
                    "val_metrics": {"mean": {"rmse": 0.05, "mae": 0.04, "r2": 0.1}},
                },
                "acquired_batch": [
                    {"acquisition": 0.8, "pseudo_stability": 0.6, "pseudo_activity": 0.5},
                    {"acquisition": 0.6, "pseudo_stability": 0.7, "pseudo_activity": 0.4},
                ],
            }
        ]
    }
    rows = active_learning_round_rows(payload)
    assert len(rows) == 1
    assert rows[0]["round"] == 0
    assert rows[0]["acquired_count"] == 2
    assert abs(rows[0]["val_rmse_mean"] - 0.05) < 1e-8


def test_active_learning_acquisition_rows_flattens_batches() -> None:
    payload = {
        "rounds": [
            {
                "round": 1,
                "acquired_batch": [
                    {"sequence": "AAA", "acquisition": 0.9},
                    {"sequence": "AAT", "acquisition": 0.8},
                ],
            }
        ]
    }
    rows = active_learning_acquisition_rows(payload)
    assert len(rows) == 2
    assert rows[0]["round"] == 1
    assert rows[1]["sequence"] == "AAT"
