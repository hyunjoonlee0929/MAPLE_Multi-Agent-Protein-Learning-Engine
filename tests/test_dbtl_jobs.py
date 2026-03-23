from __future__ import annotations

from core.dbtl_jobs import build_dbtl_ingest_command


def test_build_dbtl_ingest_command_contains_required_args() -> None:
    cmd = build_dbtl_ingest_command(
        seed_data="data/sample_property_labels.csv",
        dbtl_input="data/sample_dbtl_results.csv",
        dbtl_format="auto",
        output_dir="outputs/dbtl_ingest",
        checkpoint_out="checkpoints/property_linear_dbtl.npz",
        embedding_dim=128,
        val_ratio=0.2,
        split_seed=42,
        ridge_alphas="1e-4,1e-3",
        min_imported_records=1,
    )
    assert "scripts/dbtl_ingest_retrain.py" in cmd
    assert "--dbtl-input" in cmd
    assert "data/sample_dbtl_results.csv" in cmd
