import pytest
from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingService


@pytest.fixture(scope="module")
def vector_service():
    return VectorStoreService()


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_indexing(vector_service, embedding_service):
    """Test indexing chunks into ChromaDB with proper metadata."""
    chunks = [
        {
            "chunk_id": "doc_101_p1_c1",
            "text": "[Document: TestDocA | Page 1 | Section: CAN Comm]\nCanGatewaySWC manages CAN frame routing.",
            "metadata": {
                "document_id": "101",
                "document_name": "TestDocA.pdf",
                "page": 1,
                "section": "CAN Comm",
                "chunk_id": "doc_101_p1_c1",
                "source_type": "text"
            }
        },
        {
            "chunk_id": "doc_101_p2_c2",
            "text": "[Document: TestDocA | Page 2 | Section: Battery]\nBatteryManagementSWC reports battery state of charge.",
            "metadata": {
                "document_id": "101",
                "document_name": "TestDocA.pdf",
                "page": 2,
                "section": "Battery",
                "chunk_id": "doc_101_p2_c2",
                "source_type": "text"
            }
        }
    ]
    texts = [c["text"] for c in chunks]
    embs = embedding_service.embed_batch(texts)
    added = vector_service.add_chunks(document_id=101, chunks=chunks, embeddings=embs)
    assert added == 2


def test_retrieval(vector_service, embedding_service):
    """Test retrieving top-k chunks with cosine similarity."""
    query = "How does CAN frame routing work?"
    q_emb = embedding_service.embed_text(query)
    results = vector_service.search(document_id=101, query_embedding=q_emb, top_k=2)

    assert len(results) > 0
    top_chunk = results[0]
    assert "CanGatewaySWC" in top_chunk["content"] or "CAN Comm" in top_chunk["section"]
    assert top_chunk["similarity"] >= 0.0


def test_document_isolation(vector_service, embedding_service):
    """
    Security requirement: Test that querying document A cannot retrieve chunks from document B.
    """
    # Index document 202 with confidential powertrain details
    chunks_b = [
        {
            "chunk_id": "doc_202_p1_c1",
            "text": "[Document: TestDocB | Page 1 | Section: Powertrain]\nConfidentialMotorInverter operates at 400V PWM.",
            "metadata": {
                "document_id": "202",
                "document_name": "TestDocB.pdf",
                "page": 1,
                "section": "Powertrain",
                "chunk_id": "doc_202_p1_c1",
                "source_type": "text"
            }
        }
    ]
    embs_b = embedding_service.embed_batch([c["text"] for c in chunks_b])
    vector_service.add_chunks(document_id=202, chunks=chunks_b, embeddings=embs_b)

    # Querying Document 101 for 'ConfidentialMotorInverter' must return NO chunks from Doc 202
    q_emb = embedding_service.embed_text("ConfidentialMotorInverter")
    results_for_doc_101 = vector_service.search(document_id=101, query_embedding=q_emb, top_k=5)

    for item in results_for_doc_101:
        assert item["document_id"] == 101, f"Data leak! Found chunk belonging to doc {item['document_id']}"
        assert "ConfidentialMotorInverter" not in item["content"]

    # Verify querying Document 202 specifically does find it
    results_for_doc_202 = vector_service.search(document_id=202, query_embedding=q_emb, top_k=5)
    assert len(results_for_doc_202) == 1
    assert results_for_doc_202[0]["document_id"] == 202

    # Cleanup test documents
    vector_service.delete_by_document(101)
    vector_service.delete_by_document(202)
