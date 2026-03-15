from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Question:
    id: str
    bucket: str
    question: str
    gold_answer: Optional[str] = None

@dataclass
class AnswerRecord:
    run_id: str
    question_id: str
    bucket: str
    question: str
    system_name: str
    answer: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
