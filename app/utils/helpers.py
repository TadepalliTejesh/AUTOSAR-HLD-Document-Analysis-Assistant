import hashlib
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any


def calculate_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file on disk."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def calculate_bytes_sha256(data: bytes) -> str:
    """Calculate SHA-256 hash of byte buffer."""
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to avoid path traversal and invalid characters."""
    base = Path(filename).name
    sanitized = re.sub(r"[^\w\s\.-]", "", base).strip()
    sanitized = re.sub(r"\s+", "_", sanitized)
    if not sanitized.lower().endswith(".pdf"):
        sanitized += ".pdf"
    return sanitized or "document.pdf"


def evaluate_confidence(
    max_similarity: float,
    supporting_chunks_count: int,
    min_relevance_threshold: float = 0.50
) -> Tuple[str, str]:
    """
    Explainable confidence indicator derived from vector similarity and evidence count.
    Calibrated for dense sentence-transformers (BAAI/bge-small-en-v1.5).
    Levels: High, Medium, Low, Insufficient
    """
    if max_similarity < min_relevance_threshold or supporting_chunks_count == 0:
        return "Insufficient", f"Retrieval score {max_similarity:.3f} below threshold {min_relevance_threshold:.2f} (Supporting chunks: {supporting_chunks_count})"
    
    if max_similarity >= 0.70 and supporting_chunks_count >= 2:
        return "High", f"Strong semantic alignment ({max_similarity:.3f}) across {supporting_chunks_count} supporting sections"
    elif max_similarity >= 0.58 and supporting_chunks_count >= 1:
        return "Medium", f"Moderate retrieval evidence ({max_similarity:.3f}) with {supporting_chunks_count} relevant chunks"
    elif max_similarity >= min_relevance_threshold:
        return "Low", f"Weak retrieval evidence ({max_similarity:.3f}) near relevance threshold"
    else:
        return "Insufficient", f"No reliable supporting evidence found ({max_similarity:.3f})"
