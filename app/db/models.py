import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey
)
from sqlalchemy.orm import relationship
from app.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    sha256 = Column(String(64), index=True, nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    sections_count = Column(Integer, default=0)
    chunks_count = Column(Integer, default=0)
    entities_count = Column(Integer, default=0)
    status = Column(String(50), default="PENDING")  # PENDING, PROCESSING, INDEXED, FAILED
    error_message = Column(Text, nullable=True)
    version = Column(Integer, default=1)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="documents")
    entities = relationship("ArchitectureEntity", back_populates="document", cascade="all, delete-orphan")
    inconsistencies = relationship("PotentialInconsistency", back_populates="document", cascade="all, delete-orphan")
    queries = relationship("QueryHistory", back_populates="document", cascade="all, delete-orphan")
    audit_events = relationship("AuditEvent", back_populates="document", cascade="all, delete-orphan")


class ArchitectureEntity(Base):
    __tablename__ = "architecture_entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    entity_type = Column(String(50), nullable=False)  # COMPONENT, INTERFACE, PORT, SIGNAL, DEPENDENCY
    name = Column(String(255), nullable=False, index=True)
    details = Column(Text, nullable=True)  # JSON-encoded details
    page = Column(Integer, nullable=False)
    section = Column(String(255), nullable=False)
    evidence = Column(Text, nullable=False)

    document = relationship("Document", back_populates="entities")


class PotentialInconsistency(Base):
    __tablename__ = "potential_inconsistencies"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    entity_name = Column(String(255), nullable=False, index=True)
    issue_type = Column(String(100), nullable=False)
    evidence_1 = Column(Text, nullable=False)
    evidence_2 = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(100), default="Requires Engineer Review")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="inconsistencies")


class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(String(50), nullable=False)  # High, Medium, Low, Insufficient
    similarity_score = Column(Float, default=0.0)
    citations = Column(Text, nullable=True)  # JSON list
    retrieved_chunks = Column(Text, nullable=True)  # JSON list
    execution_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="queries")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    event_type = Column(String(100), nullable=False)  # DOCUMENT_UPLOADED, DOCUMENT_INDEXED, etc.
    details = Column(Text, nullable=True)  # JSON
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="audit_events")
