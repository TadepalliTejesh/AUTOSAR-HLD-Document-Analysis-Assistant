import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.db.database import get_db
from app.db.repositories import (
    ProjectRepository,
    DocumentRepository,
    EntityRepository,
    InconsistencyRepository,
    AuditRepository
)
from app.schemas.document import DocumentResponse, IngestionProgressResponse
from app.utils.helpers import calculate_sha256, sanitize_filename
from app.services.pdf_parser import PDFParser, PDFParserError
from app.services.chunker import SectionAwareChunker
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store_service
from app.services.entity_extractor import entity_extractor
from app.services.inconsistency_detector import inconsistency_detector

router = APIRouter(tags=["Documents"])


@router.get("/documents", response_model=List[DocumentResponse])
def list_documents(
    project_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List all ingested AUTOSAR HLD documents."""
    return DocumentRepository.list_all(db, project_id=project_id)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Retrieve document metadata by ID."""
    doc = DocumentRepository.get_by_id(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )
    return doc


@router.post("/upload-hld", response_model=IngestionProgressResponse)
async def upload_hld(
    file: UploadFile = File(...),
    project_name: str = Form(default="AUTOSAR HLD Pilot"),
    db: Session = Depends(get_db)
):
    """
    Upload and ingest an AUTOSAR HLD PDF.
    Extracts text, headings, tables, creates section-aware chunks,
    generates embeddings, extracts architecture entities, and detects inconsistencies.
    """
    # 1. Validate file extension
    original_filename = file.filename or "document.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .pdf files are accepted."
        )

    # 2. Sanitize filename and save temporarily
    clean_name = sanitize_filename(original_filename)
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"tmp_{clean_name}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file."
        )

    # 3. Calculate SHA-256 hash
    sha256 = calculate_sha256(str(temp_path))
    file_size = os.path.getsize(temp_path)

    # 4. Check for duplicate upload
    existing_doc = DocumentRepository.get_by_hash(db, sha256)
    if existing_doc and existing_doc.status == "INDEXED":
        # Clean temporary file
        if temp_path.exists():
            temp_path.unlink()
        return IngestionProgressResponse(
            document_id=existing_doc.id,
            document_name=existing_doc.filename,
            sha256=existing_doc.sha256,
            pages=existing_doc.page_count,
            sections_detected=existing_doc.sections_count,
            chunks_created=existing_doc.chunks_count,
            embeddings_generated=existing_doc.chunks_count,
            entities_detected=existing_doc.entities_count,
            status="Indexed ✓ (Existing)",
            message="Document already indexed. Returning existing knowledge base."
        )

    # Move to permanent location
    final_path = upload_dir / f"{sha256[:12]}_{clean_name}"
    shutil.move(str(temp_path), str(final_path))

    # 5. Create Project & Document in SQLite
    project = ProjectRepository.get_or_create(db, name=project_name)
    version = (existing_doc.version + 1) if existing_doc else 1
    doc_record = DocumentRepository.create(
        db=db,
        project_id=project.id,
        filename=clean_name,
        original_filename=original_filename,
        sha256=sha256,
        file_path=str(final_path),
        file_size=file_size,
        version=version
    )

    AuditRepository.log(
        db=db,
        event_type="DOCUMENT_UPLOADED",
        document_id=doc_record.id,
        details={"filename": clean_name, "sha256": sha256, "size": file_size}
    )

    try:
        # 6. Parse PDF
        parser = PDFParser(str(final_path))
        parsed_doc = parser.parse()
        pages_count = parsed_doc["page_count"]
        sections_count = parsed_doc["sections_count"]

        # 7. Section-aware chunking
        chunker = SectionAwareChunker()
        chunks = chunker.chunk_document(
            document_id=doc_record.id,
            document_name=clean_name,
            parsed_pages=parsed_doc["pages"]
        )

        if not chunks:
            raise PDFParserError("No meaningful chunks could be extracted from the document.")

        # 8. Generate Embeddings locally
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)

        # 9. Store embeddings in ChromaDB
        vector_store_service.add_chunks(
            document_id=doc_record.id,
            chunks=chunks,
            embeddings=embeddings
        )

        # 10. Extract Architecture Entities
        entities = entity_extractor.extract_from_pages(parsed_doc["pages"])
        EntityRepository.save_entities(db, doc_record.id, entities)

        # 11. Detect Potential Inconsistencies
        inconsistencies = inconsistency_detector.analyze(entities)
        InconsistencyRepository.save_inconsistencies(db, doc_record.id, inconsistencies)

        # 12. Update Database Record
        DocumentRepository.update_indexing_stats(
            db=db,
            doc_id=doc_record.id,
            page_count=pages_count,
            sections_count=sections_count,
            chunks_count=len(chunks),
            entities_count=len(entities),
            status="INDEXED"
        )

        # 13. Audit Event
        AuditRepository.log(
            db=db,
            event_type="DOCUMENT_INDEXED",
            document_id=doc_record.id,
            details={
                "pages": pages_count,
                "sections": sections_count,
                "chunks": len(chunks),
                "entities": len(entities),
                "inconsistencies": len(inconsistencies)
            }
        )

        return IngestionProgressResponse(
            document_id=doc_record.id,
            document_name=clean_name,
            sha256=sha256,
            pages=pages_count,
            sections_detected=sections_count,
            chunks_created=len(chunks),
            embeddings_generated=len(embeddings),
            entities_detected=len(entities),
            status="Indexed ✓",
            message="Document successfully processed and indexed."
        )

    except PDFParserError as pe:
        logger.error(f"PDF parsing error for doc {doc_record.id}: {pe}")
        DocumentRepository.update_indexing_stats(
            db=db,
            doc_id=doc_record.id,
            page_count=0,
            sections_count=0,
            chunks_count=0,
            entities_count=0,
            status="FAILED",
            error_message=str(pe)
        )
        AuditRepository.log(
            db=db,
            event_type="DOCUMENT_FAILED",
            document_id=doc_record.id,
            details={"error": str(pe)}
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF Parsing Failed: {str(pe)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error indexing doc {doc_record.id}: {e}")
        DocumentRepository.update_indexing_stats(
            db=db,
            doc_id=doc_record.id,
            page_count=0,
            sections_count=0,
            chunks_count=0,
            entities_count=0,
            status="FAILED",
            error_message=str(e)
        )
        AuditRepository.log(
            db=db,
            event_type="DOCUMENT_FAILED",
            document_id=doc_record.id,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during document indexing. Check system logs."
        )
