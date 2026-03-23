"""Attach external validation report metadata to MAPLE artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_validation_metadata(
    root: Path,
    leaderboard_path_text: str | None,
    cv_report_path_text: str | None,
) -> dict | None:
    leaderboard_path = None
    if leaderboard_path_text:
        p = Path(leaderboard_path_text).expanduser()
        leaderboard_path = p if p.is_absolute() else (root / p)

    cv_report_path = None
    if cv_report_path_text:
        p = Path(cv_report_path_text).expanduser()
        cv_report_path = p if p.is_absolute() else (root / p)

    leaderboard_payload = _load_json(leaderboard_path) if leaderboard_path else None
    cv_payload = _load_json(cv_report_path) if cv_report_path else None

    if leaderboard_payload is None and cv_payload is None:
        return None

    best = (leaderboard_payload or {}).get("best", {})
    best_mean = best.get("val_metrics", {}).get("mean", {})

    cv_summary = (cv_payload or {}).get("summary", {})
    cv_rmse = cv_summary.get("val_mean_rmse", {})

    return {
        "linked_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "leaderboard": (str(leaderboard_path) if leaderboard_path else None),
            "cv_report": (str(cv_report_path) if cv_report_path else None),
        },
        "leaderboard": {
            "available": leaderboard_payload is not None,
            "num_ranked": len((leaderboard_payload or {}).get("ranked_results", [])),
            "best_checkpoint": best.get("checkpoint"),
            "best_val_rmse": _safe_float(best_mean.get("rmse")),
        },
        "cv_report": {
            "available": cv_payload is not None,
            "num_runs": len((cv_payload or {}).get("runs", [])),
            "val_rmse_mean": _safe_float(cv_rmse.get("mean")),
            "val_rmse_std": _safe_float(cv_rmse.get("std")),
        },
    }


def attach_validation_metadata(final_state: dict, metadata: dict | None) -> None:
    """Attach validation metadata to summary/history-exported state."""
    if not metadata:
        return

    final_state["validation_reports"] = metadata
    history = final_state.get("history", [])
    if not isinstance(history, list):
        return

    best_ckpt = metadata.get("leaderboard", {}).get("best_checkpoint")
    best_rmse = metadata.get("leaderboard", {}).get("best_val_rmse")
    cv_mean = metadata.get("cv_report", {}).get("val_rmse_mean")
    cv_std = metadata.get("cv_report", {}).get("val_rmse_std")

    for row in history:
        if not isinstance(row, dict):
            continue
        row["validation_linked"] = True
        row["validation_best_checkpoint"] = best_ckpt
        row["validation_best_val_rmse"] = best_rmse
        row["validation_cv_rmse_mean"] = cv_mean
        row["validation_cv_rmse_std"] = cv_std
