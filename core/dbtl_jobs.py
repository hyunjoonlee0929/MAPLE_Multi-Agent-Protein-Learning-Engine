"""Helpers for launching DBTL ingestion + retraining jobs from UI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DbtlJobResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def build_dbtl_ingest_command(
    seed_data: str,
    dbtl_input: str,
    dbtl_format: str,
    output_dir: str,
    checkpoint_out: str,
    embedding_dim: int,
    val_ratio: float,
    split_seed: int,
    ridge_alphas: str,
    min_imported_records: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/dbtl_ingest_retrain.py",
        "--seed-data",
        seed_data,
        "--dbtl-input",
        dbtl_input,
        "--dbtl-format",
        dbtl_format,
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
        "--ridge-alphas",
        ridge_alphas,
        "--min-imported-records",
        str(min_imported_records),
    ]


def run_dbtl_ingest_job(
    root: Path,
    seed_data: str,
    dbtl_input: str,
    dbtl_format: str,
    output_dir: str,
    checkpoint_out: str,
    embedding_dim: int,
    val_ratio: float,
    split_seed: int,
    ridge_alphas: str,
    min_imported_records: int,
) -> DbtlJobResult:
    cmd = build_dbtl_ingest_command(
        seed_data=seed_data,
        dbtl_input=dbtl_input,
        dbtl_format=dbtl_format,
        output_dir=output_dir,
        checkpoint_out=checkpoint_out,
        embedding_dim=embedding_dim,
        val_ratio=val_ratio,
        split_seed=split_seed,
        ridge_alphas=ridge_alphas,
        min_imported_records=min_imported_records,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return DbtlJobResult(
        command=cmd,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
