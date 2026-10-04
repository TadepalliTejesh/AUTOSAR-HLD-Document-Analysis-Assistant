from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import AuditRepository, DocumentRepository
from app.schemas.query import AuditEventResponse

router = APIRouter(tags=["Audit"])


@router.get("/documents/{document_id}/audit", response_model=List[AuditEventResponse])
def get_document_audit_log(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Retrieve immutable audit events associated with a document."""
    doc = DocumentRepository.get_by_id(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )
    return AuditRepository.list_events(db, document_id=document_id)


@router.get("/audit", response_model=List[AuditEventResponse])
def get_all_audit_logs(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Retrieve system-wide audit trail."""
    return AuditRepository.list_events(db, limit=limit)
