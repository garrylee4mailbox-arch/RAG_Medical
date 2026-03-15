from __future__ import annotations

import os, time
from typing import Any, Dict, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

class OpenAIClient:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 256, base_url: str | None = None, api_key: str | None = None) -> None:
        if OpenAI is None:
            raise ImportError("openai package not installed")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL") or None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms_total = (time.time() - t0) * 1000
        text = resp.choices[0].message.content or ""
        meta: Dict[str, Any] = {
            "latency_ms_total": latency_ms_total,
            "latency_ms_retrieve": 0.0,
            "latency_ms_generate": latency_ms_total,
        }
        return text.strip(), meta
