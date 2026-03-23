from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train_property_numpy import load_dataset, split_train_val



def test_load_dataset_reads_sequences_and_targets() -> None:
    csv_path = Path("/Users/hyunjoon/codex/MAPLE/data/sample_property_labels.csv")
    sequences, targets = load_dataset(csv_path)
    assert len(sequences) > 0
    assert targets.shape[1] == 2


def test_split_train_val_produces_non_empty_splits() -> None:
    sequences = ["AAAA", "AAAT", "AATA", "ATAA", "TAAA"]
    targets = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.6],
        ],
        dtype=np.float32,
    )
    train_seq, train_t, val_seq, val_t = split_train_val(sequences, targets, val_ratio=0.4, seed=7)
    assert len(train_seq) > 0
    assert len(val_seq) > 0
    assert train_t.shape[1] == 2
    assert val_t.shape[1] == 2
    assert len(train_seq) + len(val_seq) == len(sequences)
