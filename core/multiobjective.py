"""Helpers for building multi-objective candidate views."""

from __future__ import annotations

import numpy as np

from utils.pareto import non_dominated_sort


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_pareto_candidate_rows(final_state: dict) -> list[dict]:
    """Build candidate rows with Pareto rank/front labels."""
    sequences = final_state.get("sequences", [])
    properties = final_state.get("properties", [])
    structures = final_state.get("structures", [])
    scores = final_state.get("scores", [])

    n = min(len(sequences), len(properties), len(structures), len(scores))
    if n == 0:
        return []

    rows: list[dict] = []
    points: list[list[float]] = []
    for i in range(n):
        prop = properties[i]
        structure = structures[i]
        stability = _safe_float(prop.get("stability"))
        activity = _safe_float(prop.get("activity"))
        uncertainty = _safe_float(prop.get("uncertainty"))
        confidence = _safe_float(structure.get("confidence"))

        rows.append(
            {
                "rank": i + 1,
                "sequence": sequences[i],
                "score": _safe_float(scores[i]),
                "stability": stability,
                "activity": activity,
                "uncertainty": uncertainty,
                "structure_confidence": confidence,
            }
        )
        points.append([stability, activity, -uncertainty, confidence])

    fronts = non_dominated_sort(np.asarray(points, dtype=np.float32))
    pareto_rank = {}
    for f_idx, front in enumerate(fronts, start=1):
        for idx in front:
            pareto_rank[int(idx)] = f_idx

    for i, row in enumerate(rows):
        row["pareto_rank"] = int(pareto_rank.get(i, len(fronts) + 1))
        row["is_pareto_front"] = row["pareto_rank"] == 1

    rows.sort(key=lambda x: (x["pareto_rank"], x["rank"]))
    return rows
