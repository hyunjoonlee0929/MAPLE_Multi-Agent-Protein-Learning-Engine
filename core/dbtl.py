"""DBTL experiment record schema and dataset merge helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_DBTL_FIELDS = {"sequence", "stability", "activity"}


def _to_float(value) -> float:
    return float(value)


def validate_dbtl_record(record: dict) -> dict:
    """Validate/normalize one DBTL test record."""
    missing = [k for k in REQUIRED_DBTL_FIELDS if k not in record]
    if missing:
        raise ValueError(f"DBTL record missing required fields: {missing}")

    seq = str(record["sequence"]).strip().upper()
    if not seq:
        raise ValueError("DBTL record has empty sequence")

    normalized = {
        "experiment_id": str(record.get("experiment_id", "")).strip() or None,
        "sequence": seq,
        "stability": _to_float(record["stability"]),
        "activity": _to_float(record["activity"]),
        "source": str(record.get("source", "dbtl")).strip() or "dbtl",
        "timestamp": str(record.get("timestamp", "")).strip() or None,
        "assay": str(record.get("assay", "")).strip() or None,
        "split": str(record.get("split", "train")).strip().lower(),
    }
    if normalized["split"] not in {"train", "val"}:
        normalized["split"] = "train"
    return normalized


def load_dbtl_records_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            if not row:
                continue
            out.append(validate_dbtl_record(dict(row)))
    return out


def load_dbtl_records_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("records", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("DBTL JSON must be a list or dict with 'records'")
    return [validate_dbtl_record(dict(item)) for item in items]


def load_dbtl_records(path: Path, fmt: str = "auto") -> list[dict]:
    mode = fmt.lower().strip()
    if mode == "auto":
        mode = "json" if path.suffix.lower() == ".json" else "csv"
    if mode == "csv":
        return load_dbtl_records_csv(path)
    if mode == "json":
        return load_dbtl_records_json(path)
    raise ValueError("Unsupported DBTL format. Use: auto|csv|json")


def merge_dbtl_into_dataset(
    train_sequences: list[str],
    train_targets: np.ndarray,
    val_sequences: list[str],
    val_targets: np.ndarray,
    dbtl_records: list[dict],
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict]:
    """Merge DBTL records into train/val sets; existing sequence labels are overwritten."""
    train_map = {seq: np.asarray(train_targets[i], dtype=np.float32) for i, seq in enumerate(train_sequences)}
    val_map = {seq: np.asarray(val_targets[i], dtype=np.float32) for i, seq in enumerate(val_sequences)}

    train_added = 0
    train_updated = 0
    val_added = 0
    val_updated = 0

    for rec in dbtl_records:
        item = validate_dbtl_record(rec)
        seq = item["sequence"]
        y = np.asarray([item["stability"], item["activity"]], dtype=np.float32)
        split = item.get("split", "train")

        if split == "val":
            if seq in val_map:
                val_updated += 1
            else:
                val_added += 1
            val_map[seq] = y
            train_map.pop(seq, None)
        else:
            if seq in train_map:
                train_updated += 1
            else:
                train_added += 1
            train_map[seq] = y
            val_map.pop(seq, None)

    new_train_sequences = sorted(train_map.keys())
    new_val_sequences = sorted(val_map.keys())
    new_train_targets = np.stack([train_map[s] for s in new_train_sequences]).astype(np.float32)
    new_val_targets = np.stack([val_map[s] for s in new_val_sequences]).astype(np.float32)

    stats = {
        "imported_records": len(dbtl_records),
        "train_added": train_added,
        "train_updated": train_updated,
        "val_added": val_added,
        "val_updated": val_updated,
    }
    return new_train_sequences, new_train_targets, new_val_sequences, new_val_targets, stats
