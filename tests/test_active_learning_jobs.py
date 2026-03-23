from __future__ import annotations

from core.active_learning_jobs import build_active_learning_command


def test_build_active_learning_command_contains_script_and_args() -> None:
    cmd = build_active_learning_command(
        data_path="data/sample_property_labels.csv",
        output_dir="outputs/active_learning",
        checkpoint_out="checkpoints/property_linear_active_learning.npz",
        embedding_dim=128,
        val_ratio=0.2,
        split_seed=42,
        rounds=3,
        batch_size=4,
        pool_size=40,
        mutation_rate=1,
        beta=0.3,
        ridge_alphas="1e-4,1e-3",
        seed=42,
    )
    assert "scripts/active_learning_cycle.py" in cmd
    assert "--rounds" in cmd
    assert "3" in cmd
