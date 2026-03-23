"""Run hard vs soft constrained optimization comparison for MAPLE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark import run_constraint_mode_comparison
from main import load_config



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hard vs soft constraint modes")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--output-dir", type=str, default="outputs/constraint_compare", help="Comparison output dir")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num-iterations", type=int, default=3, help="Iterations")
    parser.add_argument("--structure-backend", type=str, default="dummy", help="Structure backend")
    parser.add_argument("--min-plddt", type=float, default=60.0, help="Min pLDDT")
    parser.add_argument("--max-pae", type=float, default=20.0, help="Max PAE")
    parser.add_argument("--min-stability", type=float, default=None, help="Min stability")
    parser.add_argument("--min-activity", type=float, default=None, help="Min activity")
    parser.add_argument("--constraint-penalty", type=float, default=0.2, help="Soft-mode penalty")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    config = load_config(config_path)
    overrides = {
        "seed": args.seed,
        "num_iterations": args.num_iterations,
        "structure_backend": args.structure_backend,
        "min_plddt": args.min_plddt,
        "max_pae": args.max_pae,
        "min_stability": args.min_stability,
        "min_activity": args.min_activity,
        "constraint_penalty": args.constraint_penalty,
    }

    payload = run_constraint_mode_comparison(config=config, base_overrides=overrides, output_dir=output_dir)

    print("Constraint mode comparison complete.")
    print(f"Output: {output_dir}")
    print(f"Soft-Hard best score delta: {payload['delta']['best_score_soft_minus_hard']}")
    print(
        "Soft-Hard pass-rate delta: "
        f"{payload['delta']['constraint_pass_rate_soft_minus_hard']}"
    )


if __name__ == "__main__":
    main()
