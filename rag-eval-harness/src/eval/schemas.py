from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Question:
    id: str
    disease: str
    question: str
    gold_answer: Optional[str] = None
    source_group: Optional[str] = None

@dataclass
class AnswerRecord:
    run_id: str
    question_id: str
    disease: str
    question: str
    system_name: str
    answer: str
    gold_answer: Optional[str] = None
    source_group: Optional[str] = None
    answer_similarity: Optional[float] = None
    normalized_answer_for_scoring: Optional[str] = None
    normalized_gold_answer_for_scoring: Optional[str] = None
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
