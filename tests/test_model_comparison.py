import pytest
import numpy as np
from app.core.config import settings
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStoreService, get_vector_store_service


def test_default_model_configuration():
    """Verify default model is BAAI/bge-small-en-v1.5 and default collection is autosar_hld_knowledge."""
    assert settings.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert settings.DEFAULT_CHROMA_COLLECTION == "autosar_hld_knowledge"

    default_svc = EmbeddingService()
    assert default_svc.model_name == "BAAI/bge-small-en-v1.5"

    default_vstore = VectorStoreService()
    assert default_vstore.collection_name == "autosar_hld_knowledge"


def test_experimental_model_configuration():
    """Verify experimental model is configured to all-MiniLM-L6-v2 in an isolated collection."""
    assert settings.EXPERIMENTAL_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.EXPERIMENTAL_CHROMA_COLLECTION == "autosar_hld_knowledge_minilm"

    exp_emb_svc = get_embedding_service(settings.EXPERIMENTAL_EMBEDDING_MODEL)
    assert exp_emb_svc.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    exp_vstore = get_vector_store_service(settings.EXPERIMENTAL_CHROMA_COLLECTION)
    assert exp_vstore.collection_name == "autosar_hld_knowledge_minilm"

    # Ensure default singleton remains completely unmutated
    assert EmbeddingService().model_name == "BAAI/bge-small-en-v1.5"
    assert VectorStoreService().collection_name == "autosar_hld_knowledge"


def test_vector_collection_isolation_no_crosstalk():
    """Verify strict isolation between collections: no crosstalk, independent storage and search."""
    coll_a_name = "test_isolation_bge_coll"
    coll_b_name = "test_isolation_minilm_coll"

    vstore_a = get_vector_store_service(coll_a_name)
    vstore_b = get_vector_store_service(coll_b_name)

    assert vstore_a.collection_name == coll_a_name
    assert vstore_b.collection_name == coll_b_name
    assert vstore_a is not vstore_b

    # Dummy chunk for collection A
    chunk_a = [{
        "chunk_id": "test_chunk_a_1",
        "text": "Payload unique to collection A: BGE specific data",
        "metadata": {"document_id": "999", "chunk_id": "test_chunk_a_1", "page": 1, "section": "SecA"}
    }]
    emb_a = [[0.1] * 384]

    # Dummy chunk for collection B
    chunk_b = [{
        "chunk_id": "test_chunk_b_1",
        "text": "Payload unique to collection B: MiniLM specific data",
        "metadata": {"document_id": "999", "chunk_id": "test_chunk_b_1", "page": 2, "section": "SecB"}
    }]
    emb_b = [[-0.1] * 384]

    try:
        vstore_a.add_chunks(999, chunk_a, emb_a)
        vstore_b.add_chunks(999, chunk_b, emb_b)

        # Verify collection A only contains chunk A
        results_a = vstore_a.search(document_id=999, query_embedding=[0.1] * 384, top_k=5)
        assert len(results_a) == 1
        assert results_a[0]["chunk_id"] == "test_chunk_a_1"
        assert "SecA" in results_a[0]["section"]

        # Verify collection B only contains chunk B
        results_b = vstore_b.search(document_id=999, query_embedding=[-0.1] * 384, top_k=5)
        assert len(results_b) == 1
        assert results_b[0]["chunk_id"] == "test_chunk_b_1"
        assert "SecB" in results_b[0]["section"]

        # Cross-search must never leak
        leak_check_a = vstore_a.search(document_id=999, query_embedding=[-0.1] * 384, top_k=5)
        assert all(r["chunk_id"] != "test_chunk_b_1" for r in leak_check_a)

    finally:
        # Clean up test collections
        vstore_a.delete_by_document(999)
        vstore_b.delete_by_document(999)


def test_model_fallback_handling():
    """Verify that when an unavailable model is configured, EmbeddingService uses deterministic fallback gracefully."""
    fallback_svc = get_embedding_service("nonexistent/invalid-model-name-xyz-404")
    assert fallback_svc._model is None

    # Deterministic fallback vectors
    text = "Testing fallback vector generation"
    vec = fallback_svc.embed_text(text)

    assert isinstance(vec, list)
    assert len(vec) == 384
    # Check normalized
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0

    # Deterministic reproducibility
    vec_repeat = fallback_svc.embed_text(text)
    assert vec == vec_repeat
