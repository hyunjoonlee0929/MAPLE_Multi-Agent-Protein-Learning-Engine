"""Optimization agent for evolutionary sequence improvement."""

from __future__ import annotations

import random

from utils.diversity import select_diverse_sequences
from utils.mutation import random_mutation


class OptimizationAgent:
    """Selects top candidates and generates next iteration sequences."""

    def __init__(self, random_seed: int = 19) -> None:
        self.random_seed = random_seed

    def _select_elites(self, state: dict, top_k: int, strategy: str, min_distance: int) -> list[str]:
        sequences = state.get("sequences", [])
        if not sequences:
            return []

        if strategy == "diverse" and min_distance > 0:
            return select_diverse_sequences(sequences, top_k=top_k, min_distance=min_distance)

        return sequences[: max(1, min(top_k, len(sequences)))]

    def run(self, state: dict) -> dict:
        config = state.get("config", {})
        top_k = int(config.get("top_k", 3))
        num_candidates = int(config.get("num_candidates", 8))
        mutation_rate = int(config.get("mutation_rate", 1))
        min_distance = int(config.get("min_hamming_distance", 0))
        strategy = str(config.get("selection_strategy", "elitist")).strip().lower()
        iteration = int(state.get("iteration", 0))

        sequences = state.get("sequences", [])
        if not sequences:
            state["next_sequences"] = []
            return state

        elites = self._select_elites(
            state=state,
            top_k=max(1, min(top_k, len(sequences))),
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
