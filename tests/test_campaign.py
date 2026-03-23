from __future__ import annotations

import numpy as np

from core.campaign import append_labeled_records, select_novel_top_sequences


def test_select_novel_top_sequences_respects_order_and_uniqueness() -> None:
    ranked = ["A", "B", "C", "D"]
    existing = {"B", "X"}
    selected = select_novel_top_sequences(ranked, existing, batch_size=2)
    assert selected == ["A", "C"]


def test_append_labeled_records_appends_rows() -> None:
    seqs = ["A", "B"]
    targets = np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    records = [{"sequence": "C", "stability": 0.4, "activity": 0.5}]
    new_seqs, new_targets = append_labeled_records(seqs, targets, records)
    assert new_seqs[-1] == "C"
    assert new_targets.shape == (3, 2)
    assert abs(float(new_targets[-1, 0]) - 0.4) < 1e-8
