"""Sequence diversity helpers for evolutionary optimization."""

from __future__ import annotations



def hamming_distance(a: str, b: str) -> int:
    """Compute Hamming distance with length mismatch penalty."""
    min_len = min(len(a), len(b))
    dist = sum(1 for i in range(min_len) if a[i] != b[i])
    return dist + abs(len(a) - len(b))



def select_diverse_sequences(
    ranked_sequences: list[str],
    top_k: int,
    min_distance: int,
) -> list[str]:
    """Greedy selection preserving rank while enforcing sequence diversity."""
    if top_k <= 0:
        return []

    selected: list[str] = []
    for seq in ranked_sequences:
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(seq)
            continue
        if all(hamming_distance(seq, chosen) >= min_distance for chosen in selected):
            selected.append(seq)

    # Backfill with best-ranked sequences when diversity constraint is too strict.
    if len(selected) < top_k:
        for seq in ranked_sequences:
            if len(selected) >= top_k:
                break
            if seq not in selected:
                selected.append(seq)

    return selected
