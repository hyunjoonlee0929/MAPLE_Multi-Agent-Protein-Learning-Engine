from __future__ import annotations

import json
from pathlib import Path

from core.experiment_tracking import attach_validation_metadata, build_validation_metadata


def test_build_validation_metadata_from_files(tmp_path: Path) -> None:
    leaderboard_path = tmp_path / "validation_leaderboard.json"
    cv_path = tmp_path / "property_cv_report.json"

    leaderboard_path.write_text(
        json.dumps(
            {
                "ranked_results": [{"checkpoint": "a.npz"}],
                "best": {
                    "checkpoint": "a.npz",
                    "val_metrics": {"mean": {"rmse": 0.12}},
                },
            }
        ),
        encoding="utf-8",
    )
    cv_path.write_text(
        json.dumps(
            {
                "runs": [{"split_seed": 1}, {"split_seed": 7}],
                "summary": {"val_mean_rmse": {"mean": 0.2, "std": 0.03}},
            }
        ),
        encoding="utf-8",
    )

    meta = build_validation_metadata(tmp_path, str(leaderboard_path), str(cv_path))
    assert meta is not None
    assert meta["leaderboard"]["best_checkpoint"] == "a.npz"
    assert abs(float(meta["leaderboard"]["best_val_rmse"]) - 0.12) < 1e-8
    assert int(meta["cv_report"]["num_runs"]) == 2


def test_attach_validation_metadata_updates_history() -> None:
    state = {"history": [{"iteration": 0}, {"iteration": 1}]}
    meta = {
        "leaderboard": {"best_checkpoint": "best.npz", "best_val_rmse": 0.1},
        "cv_report": {"val_rmse_mean": 0.2, "val_rmse_std": 0.01},
    }
    attach_validation_metadata(state, meta)
    assert state["validation_reports"] == meta
    assert state["history"][0]["validation_linked"] is True
    assert state["history"][1]["validation_best_checkpoint"] == "best.npz"
