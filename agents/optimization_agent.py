"""Optimization agent for evolutionary sequence improvement."""

from __future__ import annotations

import random

from utils.diversity import select_diverse_sequences
from utils.mutation import random_mutation


class OptimizationAgent:
    """Selects top candidates and generates next iteration sequences."""

    def __init__(self, random_seed: int = 19) -> None:
        self.random_seed = random_seed

    def _as_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _constraint_violations(self, prop: dict, structure: dict, config: dict) -> dict[str, float]:
        violations: dict[str, float] = {}

        min_stability = self._as_float(config.get("min_stability"))
        min_activity = self._as_float(config.get("min_activity"))
        min_structure_confidence = self._as_float(config.get("min_structure_confidence"))
        min_plddt = self._as_float(config.get("min_plddt"))
        min_ptm = self._as_float(config.get("min_ptm"))
        max_pae = self._as_float(config.get("max_pae"))

        stability = self._as_float(prop.get("stability"))
        activity = self._as_float(prop.get("activity"))
        confidence = self._as_float(structure.get("confidence"))
        plddt = self._as_float(structure.get("plddt_mean"))
        ptm = self._as_float(structure.get("ptm"))
        pae = self._as_float(structure.get("pae_mean"))

        if min_stability is not None and (stability is None or stability < min_stability):
            violations["min_stability"] = 1.0 if stability is None else float(min_stability - stability)
        if min_activity is not None and (activity is None or activity < min_activity):
            violations["min_activity"] = 1.0 if activity is None else float(min_activity - activity)
        if min_structure_confidence is not None and (confidence is None or confidence < min_structure_confidence):
            violations["min_structure_confidence"] = (
                1.0 if confidence is None else float(min_structure_confidence - confidence)
            )
        if min_plddt is not None and (plddt is None or plddt < min_plddt):
            violations["min_plddt"] = 1.0 if plddt is None else float(min_plddt - plddt)
        if min_ptm is not None and (ptm is None or ptm < min_ptm):
            violations["min_ptm"] = 1.0 if ptm is None else float(min_ptm - ptm)
        if max_pae is not None and (pae is None or pae > max_pae):
            violations["max_pae"] = 1.0 if pae is None else float(pae - max_pae)

        return violations

    def _select_elites(self, ranked_sequences: list[str], top_k: int, strategy: str, min_distance: int) -> list[str]:
        if not ranked_sequences:
            return []

        if strategy == "diverse" and min_distance > 0:
            return select_diverse_sequences(ranked_sequences, top_k=top_k, min_distance=min_distance)

        return ranked_sequences[: max(1, min(top_k, len(ranked_sequences)))]

    def run(self, state: dict) -> dict:
        config = state.get("config", {})
        top_k = int(config.get("top_k", 3))
        num_candidates = int(config.get("num_candidates", 8))
        mutation_rate = int(config.get("mutation_rate", 1))
        min_distance = int(config.get("min_hamming_distance", 0))
        strategy = str(config.get("selection_strategy", "elitist")).strip().lower()
        constraint_enabled = bool(config.get("constraint_enabled", False))
        constraint_mode = str(config.get("constraint_mode", "hard")).strip().lower()
        constraint_penalty = float(config.get("constraint_penalty", 0.20))
        iteration = int(state.get("iteration", 0))

        sequences = state.get("sequences", [])
        structures = state.get("structures", [])
        properties = state.get("properties", [])
        scores = state.get("scores", [])

        if not sequences:
            state["next_sequences"] = []
            return state

        ranked_aligned = list(zip(sequences, structures, properties, scores))
        passed: list[tuple[str, float]] = []
        all_ranked: list[tuple[str, float, dict[str, float]]] = []
        violation_counts: dict[str, int] = {
            "min_stability": 0,
            "min_activity": 0,
            "min_structure_confidence": 0,
            "min_plddt": 0,
            "min_ptm": 0,
            "max_pae": 0,
        }

        for seq, structure, prop, score in ranked_aligned:
            violations = self._constraint_violations(prop, structure, config) if constraint_enabled else {}
            for k in violations:
                violation_counts[k] += 1

            if not violations:
                passed.append((seq, float(score)))

            total_violation = sum(violations.values())
            penalized = float(score) - constraint_penalty * total_violation
            all_ranked.append((seq, penalized, violations))

        if constraint_enabled and constraint_mode == "hard":
            candidate_ranked_sequences = [seq for seq, _score in passed] if passed else list(sequences)
        elif constraint_enabled and constraint_mode == "soft":
            sorted_soft = sorted(all_ranked, key=lambda x: x[1], reverse=True)
            candidate_ranked_sequences = [seq for seq, _penalized, _viol in sorted_soft]
        else:
            candidate_ranked_sequences = list(sequences)

        state["constraint_summary"] = {
            "enabled": constraint_enabled,
            "mode": constraint_mode,
            "penalty": constraint_penalty,
            "passed": len(passed),
            "total": len(sequences),
            "violation_counts": violation_counts,
        }

        elites = self._select_elites(
            ranked_sequences=candidate_ranked_sequences,
            top_k=max(1, min(top_k, len(candidate_ranked_sequences))),
            strategy=strategy,
            min_distance=min_distance,
        )

        rng = random.Random(self.random_seed + iteration)
        next_sequences = list(elites)
        while len(next_sequences) < num_candidates:
            parent = rng.choice(elites)
            child = random_mutation(parent, num_mutations=mutation_rate, rng=rng)
            next_sequences.append(child)

        state["next_sequences"] = next_sequences
        return state
