"""Property agent for embedding generation and property prediction."""

from __future__ import annotations

import numpy as np

from models.embedding_model import RandomEmbeddingModel
from models.property_model import PropertyPredictor


class PropertyAgent:
    """Predicts stability and activity from sequence embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        property_checkpoint: str | None = None,
        uncertainty_samples: int = 5,
        uncertainty_noise: float = 0.02,
    ) -> None:
        self.embedding_model = RandomEmbeddingModel(embedding_dim=embedding_dim)
        self.predictor = PropertyPredictor(
            embedding_dim=embedding_dim,
            checkpoint_path=property_checkpoint,
            uncertainty_samples=uncertainty_samples,
            uncertainty_noise=uncertainty_noise,
        )

    def run(self, state: dict) -> dict:
        sequences = state.get("sequences", [])
        embeddings = [self.embedding_model.encode(seq) for seq in sequences]

        if embeddings:
            batch = np.stack(embeddings).astype(np.float32)
            preds, unc = self.predictor.predict_with_uncertainty(batch)
            properties = [
                {
                    "stability": float(preds[idx][0]),
                    "activity": float(preds[idx][1]),
                    "uncertainty": float(unc[idx]),
                }
                for idx in range(len(preds))
            ]
        else:
            properties = []

        state["embeddings"] = embeddings
        state["properties"] = properties
        return state
