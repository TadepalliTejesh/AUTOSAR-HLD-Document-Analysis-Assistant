from typing import List, Optional
import numpy as np
from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        model_name = settings.EMBEDDING_MODEL
        logger.info(f"Loading SentenceTransformer embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer ({model_name}): {e}. Using deterministic fallback.")
            self._model = None

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of strings."""
        if not texts:
            return []

        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                )
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding error: {e}. Falling back to deterministic embeddings.")

        # Deterministic 384-dimensional fallback for offline/test resilience
        fallback_embs = []
        for text in texts:
            seed = sum(ord(c) for c in text[:100]) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(384)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            fallback_embs.append(vec.tolist())
        return fallback_embs


embedding_service = EmbeddingService()
