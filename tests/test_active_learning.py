from __future__ import annotations

import numpy as np

from core.active_learning import propose_active_learning_batch, synthetic_property_oracle


def test_synthetic_property_oracle_is_deterministic() -> None:
    a = synthetic_property_oracle("MKTFFV")
    b = synthetic_property_oracle("MKTFFV")
    assert abs(a["stability"] - b["stability"]) < 1e-12
    assert abs(a["activity"] - b["activity"]) < 1e-12


def test_propose_active_learning_batch_generates_unique_candidates() -> None:
    train_sequences = ["MKTFFV", "MKTFFI", "MKTFFL", "MKTFFA"]
    train_targets = np.asarray(
        [
            [0.6, 0.4],
            [0.5, 0.6],
            [0.55, 0.5],
            [0.52, 0.48],
        ],
        dtype=np.float32,
    )
    batch = propose_active_learning_batch(
        train_sequences=train_sequences,
        train_targets=train_targets,
        existing_sequences=set(train_sequences),
        embedding_dim=16,
        batch_size=3,
        pool_size=20,
        mutation_rate=1,
        beta=0.3,
        random_seed=7,
    )
    assert 0 < len(batch) <= 3
    seqs = [item["sequence"] for item in batch]
    assert len(seqs) == len(set(seqs))
    assert all(seq not in set(train_sequences) for seq in seqs)
