"""Helpers for launching active learning jobs from UI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActiveLearningJobResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def build_active_learning_command(
    data_path: str,
    output_dir: str,
    checkpoint_out: str,
    embedding_dim: int,
    val_ratio: float,
    split_seed: int,
    rounds: int,
    batch_size: int,
    pool_size: int,
    mutation_rate: int,
    beta: float,
    ridge_alphas: str,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/active_learning_cycle.py",
        "--data",
        data_path,
        "--output-dir",
        output_dir,
        "--checkpoint-out",
        checkpoint_out,
        "--embedding-dim",
        str(embedding_dim),
        "--val-ratio",
        str(val_ratio),
        "--split-seed",
        str(split_seed),
        "--rounds",
        str(rounds),
        "--batch-size",
        str(batch_size),
        "--pool-size",
        str(pool_size),
        "--mutation-rate",
        str(mutation_rate),
        "--beta",
        str(beta),
        "--ridge-alphas",
        ridge_alphas,
        "--seed",
        str(seed),
    ]


def run_active_learning_job(
    root: Path,
    data_path: str,
    output_dir: str,
    checkpoint_out: str,
    embedding_dim: int,
    val_ratio: float,
    split_seed: int,
    rounds: int,
    batch_size: int,
    pool_size: int,
    mutation_rate: int,
    beta: float,
    ridge_alphas: str,
    seed: int,
) -> ActiveLearningJobResult:
    cmd = build_active_learning_command(
        data_path=data_path,
        output_dir=output_dir,
        checkpoint_out=checkpoint_out,
        embedding_dim=embedding_dim,
        val_ratio=val_ratio,
        split_seed=split_seed,
        rounds=rounds,
        batch_size=batch_size,
        pool_size=pool_size,
        mutation_rate=mutation_rate,
        beta=beta,
        ridge_alphas=ridge_alphas,
        seed=seed,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return ActiveLearningJobResult(
        command=cmd,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
