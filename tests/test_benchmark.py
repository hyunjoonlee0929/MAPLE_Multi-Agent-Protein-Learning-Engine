from __future__ import annotations

from core.benchmark import build_constraint_mode_comparison



def test_build_constraint_mode_comparison_has_deltas() -> None:
    hard = {
        "mode": "hard",
        "best_score": 0.6,
        "constraint_pass_rate": 0.4,
    }
    soft = {
        "mode": "soft",
        "best_score": 0.7,
        "constraint_pass_rate": 0.5,
    }

    payload = build_constraint_mode_comparison(hard, soft)
    assert abs(payload["delta"]["best_score_soft_minus_hard"] - 0.1) < 1e-8
    assert abs(payload["delta"]["constraint_pass_rate_soft_minus_hard"] - 0.1) < 1e-8
