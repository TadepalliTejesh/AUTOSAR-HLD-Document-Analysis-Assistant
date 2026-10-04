import pytest
from app.db.database import SessionLocal, init_db
from app.db.repositories import DocumentRepository, ProjectRepository
from app.services.vector_store import vector_store_service
from app.services.embeddings import embedding_service
from app.services.rag import rag_service


@pytest.fixture(scope="module")
def setup_rag_test_doc():
    init_db()
    db = SessionLocal()
    project = ProjectRepository.get_or_create(db, "Test RAG Project")
    doc = DocumentRepository.create(
        db=db,
        project_id=project.id,
        filename="RAG_Test_Doc.pdf",
        original_filename="RAG_Test_Doc.pdf",
        sha256="fake_sha256_rag_test",
        file_path="./data/uploads/RAG_Test_Doc.pdf",
        file_size=1024
    )

    chunks = [
        {
            "chunk_id": f"doc_{doc.id}_p18_c1",
            "text": "[Document: RAG_Test_Doc.pdf | Page 18 | Section: Communication Architecture]\nThe VehicleControl component communicates with the CAN Interface through the CanIf interface.",
            "metadata": {
                "document_id": str(doc.id),
                "document_name": doc.filename,
                "page": 18,
                "section": "Communication Architecture",
                "chunk_id": f"doc_{doc.id}_p18_c1",
                "source_type": "text"
            }
        },
        {
            "chunk_id": f"doc_{doc.id}_p21_c2",
            "text": "[Document: RAG_Test_Doc.pdf | Page 21 | Section: Interface Mapping]\nThe CanIf interface binds PPort_CanTx of CanGatewaySWC to RPort_CanRx of VehicleControl.",
            "metadata": {
                "document_id": str(doc.id),
                "document_name": doc.filename,
                "page": 21,
                "section": "Interface Mapping",
                "chunk_id": f"doc_{doc.id}_p21_c2",
                "source_type": "text"
            }
        }
    ]

    embs = embedding_service.embed_batch([c["text"] for c in chunks])
    vector_store_service.add_chunks(doc.id, chunks, embs)
    DocumentRepository.update_indexing_stats(db, doc.id, 25, 5, 2, 2, "INDEXED")

    yield doc, db

    vector_store_service.delete_by_document(doc.id)
    db.close()


def test_grounded_answer(setup_rag_test_doc):
    doc, db = setup_rag_test_doc
    res = rag_service.answer_question(
        document_id=doc.id,
        question="Which component communicates with the CAN interface?",
        db=db
    )

    assert res.document_id == doc.id
    assert "VehicleControl" in res.answer or "CanIf" in res.answer or "CanGateway" in res.answer
    assert res.confidence in ["High", "Medium", "Low"]


def test_citations(setup_rag_test_doc):
    doc, db = setup_rag_test_doc
    res = rag_service.answer_question(
        document_id=doc.id,
        question="Which interface binds PPort_CanTx?",
        db=db
    )

    assert len(res.citations) > 0
    pages = [c.page for c in res.citations]
    assert 18 in pages or 21 in pages
    sections = [c.section for c in res.citations]
    assert any("Interface Mapping" in s or "Communication Architecture" in s for s in sections)


def test_insufficient_evidence(setup_rag_test_doc):
    doc, db = setup_rag_test_doc
    # Ask about something completely absent
    res = rag_service.answer_question(
        document_id=doc.id,
        question="What is the cryptographic quantum key exchange protocol for Satellite Uplink?",
        db=db
    )

    assert res.confidence == "Insufficient"
    assert "INSUFFICIENT EVIDENCE" in res.answer
    assert len(res.citations) == 0
