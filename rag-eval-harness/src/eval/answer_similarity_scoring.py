from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

ENGLISH_SUMMARY_PATTERNS = [
    re.compile(r"english\s+summary\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"summary\s+in\s+english\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL),
]


@dataclass(frozen=True)
class AnswerSimilarityResult:
    normalized_answer: Optional[str]
    normalized_gold_answer: Optional[str]
    raw_cosine_similarity: Optional[float]
    mapped_similarity: Optional[float]


def _strip_markdown_noise(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"`{1,3}", " ", cleaned)
    cleaned = re.sub(r"^[#>*\-\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"[•●◆■□▪▶►]+", " ", cleaned)
    return cleaned


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_english_summary(text: str) -> Optional[str]:
    if not text:
        return None

    for pattern in ENGLISH_SUMMARY_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = normalize_whitespace(_strip_markdown_noise(match.group(1)))
            return candidate or None

    english_lines: List[str] = []
    for raw_line in text.splitlines():
        line = normalize_whitespace(_strip_markdown_noise(raw_line))
        if not line:
            continue
        ascii_ratio = sum(1 for ch in line if ord(ch) < 128) / max(len(line), 1)
        if ascii_ratio >= 0.85 and re.search(r"[A-Za-z]", line):
            english_lines.append(line)
    if english_lines:
        return normalize_whitespace(" ".join(english_lines))
    return None


def normalize_text_for_scoring(text: Optional[str], *, prefer_english_summary: bool) -> Optional[str]:
    if not text:
        return None

    cleaned = _strip_markdown_noise(text)
    if prefer_english_summary:
        english_summary = extract_english_summary(cleaned)
        if english_summary:
            return english_summary
    normalized = normalize_whitespace(cleaned)
    return normalized or None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must be non-empty and have the same length")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def map_cosine_to_unit_interval(cosine_value: float) -> float:
    return max(0.0, min(1.0, (cosine_value + 1.0) / 2.0))


class OllamaEmbeddingClient:
    def __init__(self, model: str, base_url: Optional[str] = None, timeout: float = 60.0) -> None:
        self.model = model
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self._cache: Dict[str, List[float]] = {}

    def embed_text(self, text: str) -> List[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        url = f"{self.base_url}/api/embeddings"
        response = requests.post(
            url,
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Invalid embedding response from Ollama")
        vector = [float(value) for value in embedding]
        self._cache[text] = vector
        return vector


class AnswerSimilarityScorer:
    def __init__(self, embedding_client: OllamaEmbeddingClient) -> None:
        self.embedding_client = embedding_client

    def score_answer_similarity(self, answer: Optional[str], gold_answer: Optional[str]) -> AnswerSimilarityResult:
        normalized_answer = normalize_text_for_scoring(answer, prefer_english_summary=True)
        normalized_gold_answer = normalize_text_for_scoring(gold_answer, prefer_english_summary=False)

        if not normalized_answer or not normalized_gold_answer:
            return AnswerSimilarityResult(
                normalized_answer=normalized_answer,
                normalized_gold_answer=normalized_gold_answer,
                raw_cosine_similarity=None,
                mapped_similarity=None,
            )

        answer_embedding = self.embedding_client.embed_text(normalized_answer)
        gold_embedding = self.embedding_client.embed_text(normalized_gold_answer)
        raw_cosine = cosine_similarity(answer_embedding, gold_embedding)
        mapped_similarity = map_cosine_to_unit_interval(raw_cosine)
        return AnswerSimilarityResult(
            normalized_answer=normalized_answer,
            normalized_gold_answer=normalized_gold_answer,
            raw_cosine_similarity=raw_cosine,
            mapped_similarity=mapped_similarity,
        )


def build_answer_similarity_scorer(config: Optional[Dict[str, object]]) -> Optional[AnswerSimilarityScorer]:
    cfg = dict(config or {})
    if cfg.get("enabled", True) is False:
        LOGGER.info("Answer similarity scoring disabled by config")
        return None

    provider = str(cfg.get("provider", "ollama")).lower()
    if provider != "ollama":
        raise ValueError(f"Unsupported answer scoring provider: {provider}")

    model = str(cfg.get("model", "nomic-embed-text"))
    base_url = cfg.get("base_url")
    timeout = float(cfg.get("timeout", 60.0))
    client = OllamaEmbeddingClient(model=model, base_url=str(base_url) if base_url else None, timeout=timeout)
    return AnswerSimilarityScorer(client)
