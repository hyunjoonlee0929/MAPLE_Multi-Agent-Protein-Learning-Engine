"""Run property model retraining with validation-driven model selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retraining import select_best_trial
from models.embedding_model import RandomEmbeddingModel
from scripts.train_property_numpy import (
    fit_ridge_regression,
    load_dataset,
    predict_linear,
    split_train_val,
)
from utils.metrics import evaluate_property_metrics


def parse_alpha_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("ridge alpha grid is empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPLE property retraining pipeline")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/property_retrain")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/property_linear_best.npz")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--ridge-alphas",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1",
        help="Comma-separated ridge alpha candidates",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = Path(args.checkpoint_out)
    if not checkpoint_out.is_absolute():
        checkpoint_out = ROOT / checkpoint_out
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    sequences, targets = load_dataset(data_path)
    train_sequences, train_targets, val_sequences, val_targets = split_train_val(
        sequences,
        targets,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
    )
    embedder = RandomEmbeddingModel(embedding_dim=int(args.embedding_dim))
    train_features = np.stack([embedder.encode(seq) for seq in train_sequences]).astype(np.float32)
    val_features = np.stack([embedder.encode(seq) for seq in val_sequences]).astype(np.float32)

    alpha_grid = parse_alpha_grid(args.ridge_alphas)
    trials: list[dict] = []
    best_weights = None
    best_bias = None

    for alpha in alpha_grid:
        weights, bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=float(alpha))
        val_preds = predict_linear(val_features, weights, bias)
        val_metrics = evaluate_property_metrics(val_targets, val_preds)
        trial = {
            "ridge_alpha": float(alpha),
            "val_mean_rmse": float(val_metrics["mean"]["rmse"]),
            "val_mean_mae": float(val_metrics["mean"]["mae"]),
            "val_mean_r2": float(val_metrics["mean"]["r2"]),
            "val_mean_pearson": float(val_metrics["mean"]["pearson"]),
        }
        trials.append(trial)

    best_trial = select_best_trial(trials)
    best_alpha = float(best_trial["ridge_alpha"])
    best_weights, best_bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=best_alpha)
    train_preds = predict_linear(train_features, best_weights, best_bias)
    val_preds = predict_linear(val_features, best_weights, best_bias)

    train_metrics = evaluate_property_metrics(train_targets, train_preds)
    val_metrics = evaluate_property_metrics(val_targets, val_preds)

    np.savez(
        checkpoint_out,
        model_type="numpy_linear",
        embedding_dim=np.int32(args.embedding_dim),
        weights=best_weights,
        bias=best_bias,
    )

    report = {
        "dataset": str(data_path),
        "split": {
            "train_count": int(train_targets.shape[0]),
            "val_count": int(val_targets.shape[0]),
            "val_ratio": float(args.val_ratio),
            "split_seed": int(args.split_seed),
        },
        "search_space": {"ridge_alphas": alpha_grid},
        "trials": trials,
        "best": {
            "ridge_alpha": best_alpha,
            "checkpoint": str(checkpoint_out),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
    }
    report_path = output_dir / "retrain_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint: {checkpoint_out}")
    print(f"Saved retrain report: {report_path}")
    print(f"Best ridge alpha: {best_alpha}")
    print(f"Best validation mean RMSE: {val_metrics['mean']['rmse']:.4f}")


if __name__ == "__main__":
    main()
