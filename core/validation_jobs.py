"""Helpers for running validation report generation jobs from UI/CLI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def build_validation_report_commands(
    data_path: str,
    checkpoints_csv: str,
    val_ratio: float,
    split_seed: int,
    split_seeds_csv: str,
    ridge_alphas_csv: str,
    leaderboard_output_dir: str,
    cv_output_dir: str,
) -> list[tuple[str, list[str]]]:
    evaluate_cmd = [
        "scripts/evaluate_property_checkpoints.py",
        "--data",
        data_path,
        "--checkpoints",
        checkpoints_csv,
        "--val-ratio",
        str(val_ratio),
        "--split-seed",
        str(split_seed),
        "--output-dir",
        leaderboard_output_dir,
    ]
    cv_cmd = [
        "scripts/property_cv_report.py",
        "--data",
        data_path,
        "--val-ratio",
        str(val_ratio),
        "--split-seeds",
        split_seeds_csv,
        "--ridge-alphas",
        ridge_alphas_csv,
        "--output-dir",
        cv_output_dir,
    ]
    return [("leaderboard", evaluate_cmd), ("cv_report", cv_cmd)]


def run_python_job(root: Path, name: str, script_args: list[str]) -> JobResult:
    cmd = [sys.executable, *script_args]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return JobResult(
        name=name,
        command=cmd,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def run_validation_report_jobs(
    root: Path,
    data_path: str,
    checkpoints_csv: str,
    val_ratio: float,
    split_seed: int,
    split_seeds_csv: str,
    ridge_alphas_csv: str,
    leaderboard_output_dir: str,
    cv_output_dir: str,
) -> list[JobResult]:
    commands = build_validation_report_commands(
        data_path=data_path,
        checkpoints_csv=checkpoints_csv,
        val_ratio=val_ratio,
        split_seed=split_seed,
        split_seeds_csv=split_seeds_csv,
        ridge_alphas_csv=ridge_alphas_csv,
        leaderboard_output_dir=leaderboard_output_dir,
        cv_output_dir=cv_output_dir,
    )
    results: list[JobResult] = []
    for name, script_args in commands:
        result = run_python_job(root=root, name=name, script_args=script_args)
        results.append(result)
        if not result.ok:
            break
    return results
