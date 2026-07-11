"""Pydantic request/response models for the Pathology RAG API."""

from typing import List, Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    report_name: Optional[str] = None        # single-document scope (Document Chat)
    report_names: Optional[List[str]] = None  # multi-document scope (Global Search)
    top_k: int = 5
    history: Optional[List[ChatMessage]] = None


class SourceOut(BaseModel):
    index: int
    filename: str
    page: Optional[int] = None
    line_bboxes: List[List[float]] = []
    text: str
    score: float


class ReportsResponse(BaseModel):
    reports: List[str]


class HealthResponse(BaseModel):
    status: str            # "ok" | "degraded"
    num_documents: int
    detail: Optional[str] = None


class UploadStartResponse(BaseModel):
    job_id: str
    filename: str
