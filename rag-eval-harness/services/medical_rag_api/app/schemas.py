from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    rag_ready: bool
    index_info: Optional[Dict[str, Any]] = None

class IndexRequest(BaseModel):
    data_sources: List[str]
    cache_dir: Optional[str] = None
    force_rebuild: bool = False

class IndexResponse(BaseModel):
    rag_ready: bool
    index_info: Dict[str, Any]

class Context(BaseModel):
    rank: int
    department: str = ""
    source: str = ""
    score: float
    text: str

class AnswerRequest(BaseModel):
    question: str
    top_n: Optional[int] = None
    score_threshold: Optional[float] = None

class AnswerResponse(BaseModel):
    answer: str
    contexts: List[Context]
    meta: Dict[str, Any]
