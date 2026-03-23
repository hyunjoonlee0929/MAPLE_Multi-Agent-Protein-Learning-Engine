from __future__ import annotations

from core.multiobjective import build_pareto_candidate_rows


def test_build_pareto_candidate_rows_marks_front() -> None:
    state = {
        "sequences": ["A", "B", "C", "D"],
        "scores": [0.9, 0.85, 0.8, 0.83],
        "properties": [
            {"stability": 0.9, "activity": 0.1, "uncertainty": 0.1},
            {"stability": 0.1, "activity": 0.9, "uncertainty": 0.1},
            {"stability": 0.4, "activity": 0.4, "uncertainty": 0.5},
            {"stability": 0.5, "activity": 0.5, "uncertainty": 0.1},
        ],
        "structures": [
            {"confidence": 0.9},
            {"confidence": 0.9},
            {"confidence": 0.6},
            {"confidence": 0.9},
        ],
    }
    rows = build_pareto_candidate_rows(state)
    front = [row["sequence"] for row in rows if row["is_pareto_front"]]
    assert "A" in front
    assert "B" in front
    assert "D" in front
    assert "C" not in front
