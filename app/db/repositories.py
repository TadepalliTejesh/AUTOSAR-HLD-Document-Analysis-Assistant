import json
import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.db.models import (
    Project,
    Document,
    ArchitectureEntity,
    PotentialInconsistency,
    QueryHistory,
    AuditEvent
)


class ProjectRepository:
    @staticmethod
    def get_or_create(db: Session, name: str, description: Optional[str] = None) -> Project:
        project = db.query(Project).filter(Project.name == name).first()
        if not project:
            project = Project(name=name, description=description or "AUTOSAR HLD Project")
            db.add(project)
            db.commit()
            db.refresh(project)
        return project

    @staticmethod
    def list_all(db: Session) -> List[Project]:
        return db.query(Project).all()


class DocumentRepository:
    @staticmethod
    def get_by_id(db: Session, doc_id: int) -> Optional[Document]:
        return db.query(Document).filter(Document.id == doc_id).first()

    @staticmethod
    def get_by_hash(db: Session, sha256: str) -> Optional[Document]:
        return db.query(Document).filter(Document.sha256 == sha256).first()

    @staticmethod
    def list_all(db: Session, project_id: Optional[int] = None) -> List[Document]:
        query = db.query(Document)
        if project_id:
            query = query.filter(Document.project_id == project_id)
        return query.order_by(Document.uploaded_at.desc()).all()

    @staticmethod
    def create(
        db: Session,
        project_id: int,
        filename: str,
        original_filename: str,
        sha256: str,
        file_path: str,
        file_size: int,
        version: int = 1
    ) -> Document:
        doc = Document(
            project_id=project_id,
            filename=filename,
            original_filename=original_filename,
            sha256=sha256,
            file_path=file_path,
            file_size=file_size,
            version=version,
            status="PENDING",
            uploaded_at=datetime.datetime.utcnow()
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    @staticmethod
    def update_indexing_stats(
        db: Session,
        doc_id: int,
        page_count: int,
        sections_count: int,
        chunks_count: int,
        entities_count: int,
        status: str = "INDEXED",
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.page_count = page_count
            doc.sections_count = sections_count
            doc.chunks_count = chunks_count
            doc.entities_count = entities_count
            doc.status = status
            doc.error_message = error_message
            if status == "INDEXED":
                doc.indexed_at = datetime.datetime.utcnow()
            db.commit()
            db.refresh(doc)
        return doc


class EntityRepository:
    @staticmethod
    def save_entities(db: Session, doc_id: int, entities: List[Dict[str, Any]]) -> int:
        db.query(ArchitectureEntity).filter(ArchitectureEntity.document_id == doc_id).delete()
        count = 0
        for item in entities:
            entity = ArchitectureEntity(
                document_id=doc_id,
                entity_type=item.get("entity_type", "COMPONENT"),
                name=item.get("name", "Unknown"),
                details=json.dumps(item.get("details", {})),
                page=item.get("page", 1),
                section=item.get("section", "General"),
                evidence=item.get("evidence", "")
            )
            db.add(entity)
            count += 1
        db.commit()
        return count

    @staticmethod
    def get_by_document(
        db: Session,
        doc_id: int,
        entity_type: Optional[str] = None
    ) -> List[ArchitectureEntity]:
        query = db.query(ArchitectureEntity).filter(ArchitectureEntity.document_id == doc_id)
        if entity_type:
            query = query.filter(ArchitectureEntity.entity_type == entity_type)
        return query.order_by(ArchitectureEntity.page.asc()).all()


class InconsistencyRepository:
    @staticmethod
    def save_inconsistencies(db: Session, doc_id: int, inconsistencies: List[Dict[str, Any]]) -> int:
        db.query(PotentialInconsistency).filter(PotentialInconsistency.document_id == doc_id).delete()
        count = 0
        for item in inconsistencies:
            inc = PotentialInconsistency(
                document_id=doc_id,
                entity_name=item.get("entity_name", "Unknown"),
                issue_type=item.get("issue_type", "CROSS_SECTION_CONFLICT"),
                evidence_1=item.get("evidence_1", ""),
                evidence_2=item.get("evidence_2", ""),
                reason=item.get("reason", ""),
                status=item.get("status", "Requires Engineer Review")
            )
            db.add(inc)
            count += 1
        db.commit()
        return count

    @staticmethod
    def get_by_document(db: Session, doc_id: int) -> List[PotentialInconsistency]:
        return db.query(PotentialInconsistency).filter(PotentialInconsistency.document_id == doc_id).all()


class QueryRepository:
    @staticmethod
    def log_query(
        db: Session,
        doc_id: int,
        question: str,
        answer: str,
        confidence: str,
        similarity_score: float,
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        execution_time_ms: float
    ) -> QueryHistory:
        query_entry = QueryHistory(
            document_id=doc_id,
            question=question,
            answer=answer,
            confidence=confidence,
            similarity_score=similarity_score,
            citations=json.dumps(citations),
            retrieved_chunks=json.dumps(retrieved_chunks),
            execution_time_ms=execution_time_ms,
            created_at=datetime.datetime.utcnow()
        )
        db.add(query_entry)
        db.commit()
        db.refresh(query_entry)
        return query_entry

    @staticmethod
    def get_by_document(db: Session, doc_id: int, limit: int = 50) -> List[QueryHistory]:
        return db.query(QueryHistory).filter(QueryHistory.document_id == doc_id).order_by(QueryHistory.created_at.desc()).limit(limit).all()


class AuditRepository:
    @staticmethod
    def log(
        db: Session,
        event_type: str,
        document_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        audit = AuditEvent(
            document_id=document_id,
            event_type=event_type,
            details=json.dumps(details or {}),
            timestamp=datetime.datetime.utcnow()
        )
        db.add(audit)
        db.commit()
        db.refresh(audit)
        return audit

    @staticmethod
    def list_events(db: Session, document_id: Optional[int] = None, limit: int = 100) -> List[AuditEvent]:
        query = db.query(AuditEvent)
        if document_id:
            query = query.filter(AuditEvent.document_id == document_id)
        return query.order_by(AuditEvent.timestamp.desc()).limit(limit).all()
