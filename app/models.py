from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata filters")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: Optional[str] = Field(None, description="Limit retrieval to a single document")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str
    processing_time_ms: float
    model_used: str
    tokens_used: Optional[int] = None
    grounded: bool = True


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: Optional[str] = None
    status: DocumentStatus
    message: str
    estimated_processing_time_seconds: Optional[int] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    file_name: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    chunk_count: Optional[int] = None


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
