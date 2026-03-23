"""Closed-loop in-silico campaign helpers."""

from __future__ import annotations

import numpy as np


def select_novel_top_sequences(
    ranked_sequences: list[str],
    existing_sequences: set[str],
    batch_size: int,
) -> list[str]:
    """Select top-ranked novel sequences up to batch_size."""
    out = []
    seen = set(existing_sequences)
    for seq in ranked_sequences:
        if seq in seen:
            continue
        seen.add(seq)
        out.append(seq)
        if len(out) >= batch_size:
            break
    return out


def append_labeled_records(
    train_sequences: list[str],
    train_targets: np.ndarray,
    records: list[dict],
) -> tuple[list[str], np.ndarray]:
    """Append labeled records to train set."""
    if not records:
        return list(train_sequences), np.asarray(train_targets, dtype=np.float32)

    seqs = list(train_sequences)
    targets = np.asarray(train_targets, dtype=np.float32)
    new_rows = []
    for item in records:
        seq = str(item["sequence"])
        st = float(item["stability"])
        ac = float(item["activity"])
        seqs.append(seq)
        new_rows.append([st, ac])
    appended = np.asarray(new_rows, dtype=np.float32)
    targets = np.concatenate([targets, appended], axis=0)
    return seqs, targets
