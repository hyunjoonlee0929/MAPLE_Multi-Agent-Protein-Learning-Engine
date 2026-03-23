"""Property agent for embedding generation and property prediction."""

from __future__ import annotations

import numpy as np

from models.embedding_model import build_embedding_model
from models.property_model import PropertyPredictor


class PropertyAgent:
    """Predicts stability and activity from sequence embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        embedding_backend: str = "random",
        embedding_model_id: str | None = None,
        embedding_device: str = "cpu",
        embedding_pooling: str = "mean",
        embedding_allow_mock: bool = True,
        property_checkpoint: str | None = None,
        uncertainty_samples: int = 5,
        uncertainty_noise: float = 0.02,
    ) -> None:
        self.embedding_model = build_embedding_model(
            backend=embedding_backend,
            embedding_dim=embedding_dim,
            model_id=embedding_model_id,
            device=embedding_device,
            pooling=embedding_pooling,
            allow_mock=embedding_allow_mock,
        )
        self.predictor = PropertyPredictor(
            embedding_dim=int(self.embedding_model.embedding_dim),
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
