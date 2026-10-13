import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "AUTOSAR HLD Document Analysis Assistant"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EXPERIMENTAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHROMA_COLLECTION: str = "autosar_hld_knowledge"
    EXPERIMENTAL_CHROMA_COLLECTION: str = "autosar_hld_knowledge_minilm"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    CHROMA_DIR: str = "./data/chroma"
    UPLOAD_DIR: str = "./data/uploads"
    EXPORT_DIR: str = "./data/exports"
    DATABASE_URL: str = "sqlite:///./data/autosar.db"

    TOP_K: int = 6
    MIN_RELEVANCE: float = 0.50
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

    def ensure_directories(self) -> None:
        """Ensure necessary data directories exist."""
        for d in [self.CHROMA_DIR, self.UPLOAD_DIR, self.EXPORT_DIR, "./data"]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
