from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.dbtl import load_dbtl_records, merge_dbtl_into_dataset, validate_dbtl_record


def test_validate_dbtl_record_normalizes_fields() -> None:
    rec = validate_dbtl_record(
        {
            "sequence": "mktffv",
            "stability": "0.6",
            "activity": "0.4",
            "split": "train",
        }
    )
    assert rec["sequence"] == "MKTFFV"
    assert abs(rec["stability"] - 0.6) < 1e-8
    assert rec["split"] == "train"


def test_load_dbtl_records_auto_csv(tmp_path: Path) -> None:
    p = tmp_path / "dbtl.csv"
    p.write_text(
        "sequence,stability,activity,split\nAAAA,0.5,0.4,train\n",
        encoding="utf-8",
    )
    rows = load_dbtl_records(p, fmt="auto")
    assert len(rows) == 1
    assert rows[0]["sequence"] == "AAAA"


def test_load_dbtl_records_auto_json(tmp_path: Path) -> None:
    p = tmp_path / "dbtl.json"
    p.write_text(
        json.dumps([{"sequence": "AAAT", "stability": 0.6, "activity": 0.5, "split": "val"}]),
        encoding="utf-8",
    )
    rows = load_dbtl_records(p, fmt="auto")
    assert len(rows) == 1
    assert rows[0]["split"] == "val"


def test_merge_dbtl_into_dataset_updates_and_adds() -> None:
    train_seq = ["A", "B"]
    train_t = np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    val_seq = ["C"]
    val_t = np.asarray([[0.3, 0.4]], dtype=np.float32)
    records = [
        {"sequence": "B", "stability": 0.9, "activity": 0.8, "split": "train"},
        {"sequence": "D", "stability": 0.5, "activity": 0.6, "split": "val"},
    ]
    n_train_seq, n_train_t, n_val_seq, n_val_t, stats = merge_dbtl_into_dataset(
        train_seq, train_t, val_seq, val_t, records
    )
    assert "B" in n_train_seq
    assert "D" in n_val_seq
    assert n_train_t.shape[1] == 2
    assert n_val_t.shape[1] == 2
    assert stats["train_updated"] == 1
    assert stats["val_added"] == 1
