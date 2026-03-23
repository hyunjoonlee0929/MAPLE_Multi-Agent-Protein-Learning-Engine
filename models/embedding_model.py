"""Embedding model abstraction for protein sequences."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class EmbeddingModelLike(Protocol):
    """Protocol for sequence embedding backends."""

    embedding_dim: int

    def encode(self, sequence: str) -> np.ndarray:
        """Encode one sequence to a 1D vector."""


class RandomEmbeddingModel:
    """Deterministic pseudo-random embedding model using sequence hash seed."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = int(embedding_dim)
        self.backend = "random"
        self.mode = "mock"

    def encode(self, sequence: str) -> np.ndarray:
        digest = hashlib.md5(sequence.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=(self.embedding_dim,)).astype(np.float32)


class HFProteinEmbeddingModel:
    """HuggingFace-based protein encoder wrapper (ESM2 / ProtT5)."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        pooling: str = "mean",
        max_length: int = 1024,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_id = str(model_id)
        self.pooling = str(pooling).strip().lower()
        self.max_length = int(max_length)
        self.backend = "hf"
        self.mode = "external"

        if self.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        if device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = str(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()

        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
            self.model.to(self.device)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from embedding model config")
        self.embedding_dim = int(hidden_size)

    def encode(self, sequence: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            mask = inputs.get("attention_mask")
            if mask is None:
                emb = hidden.mean(dim=1)
            else:
                mask_f = mask.unsqueeze(-1).float()
                denom = mask_f.sum(dim=1).clamp(min=1.0)
                emb = (hidden * mask_f).sum(dim=1) / denom

        return emb[0].detach().cpu().numpy().astype(np.float32)


def _default_model_id(backend: str) -> str:
    if backend == "esm2":
        return "facebook/esm2_t12_35M_UR50D"
    if backend == "prott5":
        return "Rostlab/prot_t5_xl_half_uniref50-enc"
    raise ValueError(f"Unsupported embedding backend: {backend}")


def build_embedding_model(
    backend: str = "random",
    embedding_dim: int = 128,
    model_id: str | None = None,
    device: str = "cpu",
    pooling: str = "mean",
    allow_mock: bool = True,
    max_length: int = 1024,
) -> EmbeddingModelLike:
    """Build embedding model; optionally fall back to random embeddings."""
    normalized = str(backend).strip().lower()
    if normalized == "random":
        return RandomEmbeddingModel(embedding_dim=int(embedding_dim))

    if normalized not in {"esm2", "prott5"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")

    selected_model_id = str(model_id).strip() if model_id else _default_model_id(normalized)
    try:
        return HFProteinEmbeddingModel(
            model_id=selected_model_id,
            device=device,
            pooling=pooling,
            max_length=max_length,
        )
    except Exception:
        if not allow_mock:
            raise
        fallback = RandomEmbeddingModel(embedding_dim=int(embedding_dim))
        fallback.backend = normalized
        fallback.mode = "mock"
        return fallback
