"""Scoring and normalization helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np



def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    """Min-max normalize values into [0, 1]."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmin, vmax):
        return np.ones_like(arr) * 0.5

    return (arr - vmin) / (vmax - vmin)



def combined_score(
    stability: Iterable[float],
    activity: Iterable[float],
    w_stability: float = 0.5,
    w_activity: float = 0.5,
) -> np.ndarray:
    """Compute weighted score from normalized stability and activity."""
    s_norm = minmax_normalize(stability)
    a_norm = minmax_normalize(activity)
    return w_stability * s_norm + w_activity * a_norm



def combined_score_with_uncertainty(
    stability: Iterable[float],
    activity: Iterable[float],
    uncertainty: Iterable[float],
    w_stability: float = 0.45,
    w_activity: float = 0.45,
    w_uncertainty: float = 0.10,
) -> np.ndarray:
    """Weighted score with exploration bonus from normalized uncertainty."""
    s_norm = minmax_normalize(stability)
    a_norm = minmax_normalize(activity)
    u_norm = minmax_normalize(uncertainty)
    return w_stability * s_norm + w_activity * a_norm + w_uncertainty * u_norm
