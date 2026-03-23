"""Active learning utilities for property model improvement."""

from __future__ import annotations

import hashlib
import random

import numpy as np

from models.embedding_model import RandomEmbeddingModel
from utils.mutation import random_mutation


def _fit_linear_scalar(x: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-3) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    n, d = x.shape
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)
    eye = np.eye(d + 1, dtype=np.float32)
    eye[-1, -1] = 0.0
    params = np.linalg.solve(x_aug.T @ x_aug + float(ridge_alpha) * eye, x_aug.T @ y).reshape(-1)
    return params[:-1].astype(np.float32), float(params[-1])


def _predict_linear_scalar(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32) + float(b)


def _novelty_to_train(candidates: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    if train_x.size == 0:
        return np.ones((candidates.shape[0],), dtype=np.float32)
    vals = []
    for row in candidates:
        diff = train_x - row
        nn = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
        vals.append(nn)
    out = np.asarray(vals, dtype=np.float32)
    scale = float(np.mean(out)) + 1e-6
    return out / scale


def scalar_objective(stability: np.ndarray, activity: np.ndarray) -> np.ndarray:
    stability = np.asarray(stability, dtype=np.float32)
    activity = np.asarray(activity, dtype=np.float32)
    return 0.5 * stability + 0.5 * activity


def synthetic_property_oracle(sequence: str) -> dict:
    """Deterministic pseudo-label oracle used for in-silico active learning."""
    seq = sequence.strip().upper()
    if not seq:
        return {"stability": 0.0, "activity": 0.0}

    hydrophobic = set("AILMFWVY")
    charged = set("DEKRH")
    hydro_ratio = sum(1 for c in seq if c in hydrophobic) / len(seq)
    charged_ratio = sum(1 for c in seq if c in charged) / len(seq)

    digest = hashlib.md5(seq.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    jitter = float(rng.normal(0.0, 0.02))

    stability = 0.15 + 0.70 * hydro_ratio + jitter
    activity = 0.20 + 0.65 * charged_ratio + 0.10 * (1.0 - hydro_ratio) - jitter

    return {
        "stability": float(np.clip(stability, -1.0, 1.0)),
        "activity": float(np.clip(activity, -1.0, 1.0)),
    }


def propose_active_learning_batch(
    train_sequences: list[str],
    train_targets: np.ndarray,
    existing_sequences: set[str],
    embedding_dim: int,
    batch_size: int,
    pool_size: int,
    mutation_rate: int,
    beta: float,
    random_seed: int,
) -> list[dict]:
    """Propose candidate batch via surrogate mean + novelty acquisition."""
    if not train_sequences or train_targets.shape[0] == 0 or batch_size <= 0:
        return []

    embedder = RandomEmbeddingModel(embedding_dim=embedding_dim)
    train_x = np.stack([embedder.encode(seq) for seq in train_sequences]).astype(np.float32)
    score_y = scalar_objective(train_targets[:, 0], train_targets[:, 1])
    w, b = _fit_linear_scalar(train_x, score_y, ridge_alpha=1e-3)

    parent_scores = list(zip(train_sequences, score_y.tolist()))
    parent_scores.sort(key=lambda x: x[1], reverse=True)
    parents = [seq for seq, _ in parent_scores[: max(2, min(10, len(parent_scores)))]]

    rng = random.Random(random_seed)
    pool: list[str] = []
    seen = set(existing_sequences)
    seen.update(train_sequences)
    max_trials = max(pool_size * 8, 64)
    trials = 0
    while len(pool) < pool_size and trials < max_trials:
        parent = rng.choice(parents)
        child = random_mutation(parent, num_mutations=max(1, mutation_rate), rng=rng)
        trials += 1
        if child in seen:
            continue
        seen.add(child)
        pool.append(child)

    if not pool:
        return []

    pool_x = np.stack([embedder.encode(seq) for seq in pool]).astype(np.float32)
    mean_pred = _predict_linear_scalar(pool_x, w, b)
    novelty = _novelty_to_train(pool_x, train_x)
    acquisition = mean_pred + float(beta) * novelty

    ranked_idx = np.argsort(-acquisition)
    out = []
    for idx in ranked_idx[:batch_size]:
        i = int(idx)
        out.append(
            {
                "sequence": pool[i],
                "pred_mean": float(mean_pred[i]),
                "novelty": float(novelty[i]),
                "acquisition": float(acquisition[i]),
            }
        )
    return out
