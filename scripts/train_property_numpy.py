"""Train a lightweight property model from labeled CSV and export NPZ checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.embedding_model import build_embedding_model
from utils.metrics import evaluate_property_metrics



def load_dataset(csv_path: Path) -> tuple[list[str], np.ndarray]:
    sequences: list[str] = []
    targets: list[list[float]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sequence", "stability", "activity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Dataset must contain columns: sequence, stability, activity")

        for row in reader:
            seq = str(row["sequence"]).strip()
            if not seq:
                continue
            sequences.append(seq)
            targets.append([float(row["stability"]), float(row["activity"])])

    if not sequences:
        raise ValueError("No valid rows found in dataset")

    return sequences, np.asarray(targets, dtype=np.float32)


def split_train_val(
    sequences: list[str],
    targets: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    n = len(sequences)
    train_idx, val_idx = split_indices(n=n, val_ratio=val_ratio, seed=seed)
    return split_train_val_with_indices(sequences, targets, train_idx, val_idx)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples for train/validation split")

    ratio = min(max(val_ratio, 0.0), 0.9)
    val_count = int(round(n * ratio))
    val_count = max(1, min(val_count, n - 1))

    idx = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    val_idx = idx[:val_count]
    train_idx = idx[val_count:]
    return train_idx, val_idx


def split_train_val_with_indices(
    sequences: list[str],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    n = len(sequences)
    if targets.shape[0] != n:
        raise ValueError("targets length must match sequences length")

    train_seq = [sequences[int(i)] for i in train_idx]
    val_seq = [sequences[int(i)] for i in val_idx]
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    return train_seq, train_targets, val_seq, val_targets



def fit_ridge_regression(features: np.ndarray, targets: np.ndarray, ridge_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit multivariate linear regression with L2 regularization."""
    n_samples, n_features = features.shape
    x_aug = np.concatenate([features, np.ones((n_samples, 1), dtype=np.float32)], axis=1)

    eye = np.eye(n_features + 1, dtype=np.float32)
    eye[-1, -1] = 0.0

    lhs = x_aug.T @ x_aug + ridge_alpha * eye
    rhs = x_aug.T @ targets

    params = np.linalg.solve(lhs, rhs)
    weights = params[:-1, :]
    bias = params[-1, :]
    return weights.astype(np.float32), bias.astype(np.float32)



def predict_linear(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return features @ weights + bias



def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPLE NumPy property predictor")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv", help="CSV dataset path")
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/property_linear.npz",
        help="Output NPZ checkpoint path",
    )
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--embedding-backend", type=str, default="random", help="Embedding backend: random|esm2|prott5")
    parser.add_argument("--embedding-model-id", type=str, default="", help="Optional HF model id override")
    parser.add_argument("--embedding-device", type=str, default="cpu", help="Embedding device: cpu|cuda|auto")
    parser.add_argument("--embedding-pooling", type=str, default="mean", help="Embedding pooling: mean|cls")
    parser.add_argument("--disable-embedding-mock-fallback", action="store_true", help="Disable random fallback when backend fails")
    parser.add_argument("--ridge-alpha", type=float, default=1e-3, help="L2 regularization strength")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--split-seed", type=int, default=42, help="Train/val split seed")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="outputs/property_metrics/property_train_metrics.json",
        help="Output JSON path for train/val metrics",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    metrics_path = Path(args.metrics_out)
    if not metrics_path.is_absolute():
        metrics_path = root / metrics_path

    sequences, targets = load_dataset(data_path)
    train_sequences, train_targets, val_sequences, val_targets = split_train_val(
        sequences,
        targets,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
    )

    embedder = build_embedding_model(
        backend=str(args.embedding_backend),
        embedding_dim=int(args.embedding_dim),
        model_id=(str(args.embedding_model_id).strip() or None),
        device=str(args.embedding_device),
        pooling=str(args.embedding_pooling),
        allow_mock=(not args.disable_embedding_mock_fallback),
    )
    resolved_embedding_dim = int(embedder.embedding_dim)
    train_features = np.stack([embedder.encode(seq) for seq in train_sequences]).astype(np.float32)
    val_features = np.stack([embedder.encode(seq) for seq in val_sequences]).astype(np.float32)

    weights, bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=float(args.ridge_alpha))
    train_preds = predict_linear(train_features, weights, bias)
    val_preds = predict_linear(val_features, weights, bias)

    train_metrics = evaluate_property_metrics(train_targets, train_preds)
    val_metrics = evaluate_property_metrics(val_targets, val_preds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        model_type="numpy_linear",
        embedding_dim=np.int32(resolved_embedding_dim),
        embedding_backend=np.array(str(args.embedding_backend)),
        embedding_model_id=np.array(str(args.embedding_model_id).strip()),
        embedding_pooling=np.array(str(args.embedding_pooling)),
        weights=weights,
        bias=bias,
    )

    metrics_payload = {
        "dataset": str(data_path),
        "split": {
            "train_count": int(train_targets.shape[0]),
            "val_count": int(val_targets.shape[0]),
            "val_ratio": float(args.val_ratio),
            "split_seed": int(args.split_seed),
        },
        "model": {
            "type": "numpy_linear_ridge",
            "embedding_dim": resolved_embedding_dim,
            "embedding_backend": str(args.embedding_backend),
            "embedding_model_id": str(args.embedding_model_id).strip() or None,
            "embedding_pooling": str(args.embedding_pooling),
            "ridge_alpha": float(args.ridge_alpha),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {output_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Train mean metrics: "
        f"RMSE={train_metrics['mean']['rmse']:.4f}, "
        f"MAE={train_metrics['mean']['mae']:.4f}, "
        f"R2={train_metrics['mean']['r2']:.4f}, "
        f"Pearson={train_metrics['mean']['pearson']:.4f}"
    )
    print(
        "Validation mean metrics: "
        f"RMSE={val_metrics['mean']['rmse']:.4f}, "
        f"MAE={val_metrics['mean']['mae']:.4f}, "
        f"R2={val_metrics['mean']['r2']:.4f}, "
        f"Pearson={val_metrics['mean']['pearson']:.4f}"
    )


if __name__ == "__main__":
    main()
