from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class QueryRequest(BaseModel):
    document_id: int = Field(..., description="Target document ID for query")
    question: str = Field(..., min_length=2, description="Engineering question about the HLD")


class Citation(BaseModel):
    page: int
    section: str
    source_type: str = "text"
    text_snippet: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    page: int
    section: str
    source_type: str
    content: str
    similarity: float


class QueryResponse(BaseModel):
    document_id: int
    question: str
    answer: str
    confidence: str  # High, Medium, Low, Insufficient
    similarity_score: float
    citations: List[Citation]
    retrieved_evidence: List[RetrievedChunk]
    execution_time_ms: float
    potential_inconsistency: Optional[bool] = False


class AuditEventResponse(BaseModel):
    id: int
    document_id: Optional[int]
    event_type: str
    details: str
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)
