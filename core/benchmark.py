"""Benchmark helpers for MAPLE experiment comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from main import run_maple



def _extract_run_metrics(final_state: dict, resolved: dict, mode: str) -> dict:
    history = final_state.get("history", [])
    last = history[-1] if history else {}

    return {
        "mode": mode,
        "best_score": float(final_state.get("scores", [0.0])[0]) if final_state.get("scores") else None,
        "best_sequence": final_state.get("sequences", [None])[0] if final_state.get("sequences") else None,
        "constraint_pass_rate": float(last.get("constraint_pass_rate", 0.0)),
        "constraint_passed": int(last.get("constraint_passed", 0)),
        "constraint_total": int(last.get("constraint_total", 0)),
        "num_iterations": int(resolved.get("num_iterations", 0)),
        "seed": int(resolved.get("seed", 0)),
    }



def build_constraint_mode_comparison(hard: dict, soft: dict) -> dict:
    """Build normalized comparison payload from two mode summaries."""
    hard_score = hard.get("best_score")
    soft_score = soft.get("best_score")

    score_delta = None
    if hard_score is not None and soft_score is not None:
        score_delta = float(soft_score) - float(hard_score)

    pass_rate_delta = float(soft.get("constraint_pass_rate", 0.0)) - float(
        hard.get("constraint_pass_rate", 0.0)
    )

    return {
        "hard": hard,
        "soft": soft,
        "delta": {
            "best_score_soft_minus_hard": score_delta,
            "constraint_pass_rate_soft_minus_hard": pass_rate_delta,
        },
    }



def _comparison_markdown(payload: dict) -> str:
    hard = payload["hard"]
    soft = payload["soft"]
    delta = payload["delta"]

    return (
        "# Constraint Mode Comparison\n\n"
        "## Hard Mode\n"
        f"- best_score: {hard.get('best_score')}\n"
        f"- constraint_pass_rate: {hard.get('constraint_pass_rate')}\n"
        f"- best_sequence: {hard.get('best_sequence')}\n\n"
        "## Soft Mode\n"
        f"- best_score: {soft.get('best_score')}\n"
        f"- constraint_pass_rate: {soft.get('constraint_pass_rate')}\n"
        f"- best_sequence: {soft.get('best_sequence')}\n\n"
        "## Delta (soft - hard)\n"
        f"- best_score: {delta.get('best_score_soft_minus_hard')}\n"
        f"- constraint_pass_rate: {delta.get('constraint_pass_rate_soft_minus_hard')}\n"
    )



def run_constraint_mode_comparison(
    config: dict,
    base_overrides: dict,
    output_dir: Path,
) -> dict:
    """Run hard/soft constraint comparison and write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_overrides = dict(base_overrides)
    hard_overrides["constraint_enabled"] = True
    hard_overrides["constraint_mode"] = "hard"

    soft_overrides = dict(base_overrides)
    soft_overrides["constraint_enabled"] = True
    soft_overrides["constraint_mode"] = "soft"

    hard_state, hard_resolved, _ = run_maple(
        config=config,
        overrides=hard_overrides,
        output_dir=output_dir / "hard",
    )
    soft_state, soft_resolved, _ = run_maple(
        config=config,
        overrides=soft_overrides,
        output_dir=output_dir / "soft",
    )

    hard_summary = _extract_run_metrics(hard_state, hard_resolved, mode="hard")
    soft_summary = _extract_run_metrics(soft_state, soft_resolved, mode="soft")

    payload = build_constraint_mode_comparison(hard_summary, soft_summary)

    json_path = output_dir / "constraint_comparison.json"
    md_path = output_dir / "constraint_comparison.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_comparison_markdown(payload), encoding="utf-8")

    return payload
