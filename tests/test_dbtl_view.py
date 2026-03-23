from __future__ import annotations

from core.dbtl_view import dbtl_summary_row, dbtl_trial_rows


def test_dbtl_summary_row_extracts_key_fields() -> None:
    payload = {
        "merge_stats": {"imported_records": 3, "train_added": 2, "train_updated": 1, "val_added": 0, "val_updated": 0},
        "train_size": 20,
        "val_size": 5,
        "retrain_triggered": True,
        "checkpoint": "checkpoints/property_linear_dbtl.npz",
        "fit": {
            "train_metrics": {"mean": {"rmse": 0.01}},
            "val_metrics": {"mean": {"rmse": 0.05}},
        },
    }
    row = dbtl_summary_row(payload)
    assert row["imported_records"] == 3
    assert row["retrain_triggered"] is True
    assert abs(row["val_rmse_mean"] - 0.05) < 1e-8


def test_dbtl_trial_rows_flattens_trials() -> None:
    payload = {"fit": {"trials": [{"ridge_alpha": 0.001, "val_mean_rmse": 0.1}]}}
    rows = dbtl_trial_rows(payload)
    assert len(rows) == 1
    assert abs(rows[0]["ridge_alpha"] - 0.001) < 1e-12
