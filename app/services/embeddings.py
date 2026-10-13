from typing import List, Optional, Dict
import numpy as np
from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    _instances: Dict[str, "EmbeddingService"] = {}
    _instance: Optional["EmbeddingService"] = None
    _model = None
    _model_name: str = ""

    def __new__(cls, model_name: Optional[str] = None):
        target_model = model_name or settings.EMBEDDING_MODEL
        if target_model not in cls._instances:
            instance = super(EmbeddingService, cls).__new__(cls)
            instance._model_name = target_model
            instance._init_model(target_model)
            cls._instances[target_model] = instance
            if target_model == settings.EMBEDDING_MODEL:
                cls._instance = instance
        return cls._instances[target_model]

    @property
    def model_name(self) -> str:
        return self._model_name

    def _init_model(self, model_name: str):
        logger.info(f"Loading SentenceTransformer embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info(f"Embedding model ({model_name}) loaded successfully.")
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


def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """Factory helper to obtain an EmbeddingService instance for a specified model."""
    return EmbeddingService(model_name=model_name)
