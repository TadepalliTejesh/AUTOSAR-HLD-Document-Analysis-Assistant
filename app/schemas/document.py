from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ProjectBase(BaseModel):
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")


class ProjectCreate(ProjectBase):
    pass


class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    original_filename: str
    sha256: str
    file_size: int
    page_count: int
    sections_count: int
    chunks_count: int
    entities_count: int
    status: str
    error_message: Optional[str] = None
    version: int
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


class IngestionProgressResponse(BaseModel):
    document_id: int
    document_name: str
    sha256: str
    pages: int
    sections_detected: int
    chunks_created: int
    embeddings_generated: int
    entities_detected: int
    status: str
    message: str
