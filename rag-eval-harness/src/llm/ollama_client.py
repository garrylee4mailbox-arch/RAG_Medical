from __future__ import annotations

import os, time
from typing import Any, Dict, Tuple
import requests

class OllamaClient:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 256, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        latency_ms_total = (time.time() - t0) * 1000
        meta: Dict[str, Any] = {
            "latency_ms_total": latency_ms_total,
            "latency_ms_retrieve": 0.0,
            "latency_ms_generate": latency_ms_total,
        }
        return str(data.get("response","")).strip(), meta
