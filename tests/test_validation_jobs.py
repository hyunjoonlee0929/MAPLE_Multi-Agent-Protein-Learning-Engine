from __future__ import annotations

from core.validation_jobs import build_validation_report_commands


def test_build_validation_report_commands_contains_expected_scripts() -> None:
    commands = build_validation_report_commands(
        data_path="data/sample_property_labels.csv",
        checkpoints_csv="checkpoints/a.npz,checkpoints/b.npz",
        val_ratio=0.2,
        split_seed=42,
        split_seeds_csv="1,7,13",
        ridge_alphas_csv="1e-4,1e-3",
        leaderboard_output_dir="outputs/property_validation",
        cv_output_dir="outputs/property_cv",
    )
    assert len(commands) == 2
    assert commands[0][0] == "leaderboard"
    assert commands[1][0] == "cv_report"
    assert commands[0][1][0] == "scripts/evaluate_property_checkpoints.py"
    assert commands[1][1][0] == "scripts/property_cv_report.py"
