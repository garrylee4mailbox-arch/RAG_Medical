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

    def generate(self, prompt: str, *, stream: bool = True, echo_stream: bool | None = None) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "think": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=self.timeout, stream=stream)
        r.raise_for_status()
        chunk_count = 0
        response_keys: list[str] = []
        if echo_stream is None:
            echo_stream = os.getenv("OLLAMA_STREAM_TO_CONSOLE", "").strip().lower() in {"1", "true", "yes", "y", "on"}

        if stream:
            chunks: list[str] = []
            final_event: Dict[str, Any] = {}
            # Collect streamed response chunks and concatenate them back into one answer string.
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                data = requests.models.complexjson.loads(raw_line)
                if not response_keys:
                    response_keys = list(data.keys())
                piece = str(data.get("response", ""))
                if piece:
                    chunk_count += 1
                    chunks.append(piece)
                    if echo_stream:
                        print(piece, end="", flush=True)
                final_event = data
            if echo_stream and chunk_count:
                print()
            text = "".join(chunks).strip()
            if not text:
                message = final_event.get("message")
                if isinstance(message, dict):
                    text = str(message.get("content", "")).strip()
            if not text:
                text = str(final_event.get("content", "")).strip()
            if not text:
                text = str(final_event.get("thinking", "")).strip()
        else:
            data = r.json()
            response_keys = list(data.keys())
            text = str(data.get("response", "")).strip()
            if not text:
                message = data.get("message")
                if isinstance(message, dict):
                    text = str(message.get("content", "")).strip()
            if not text:
                text = str(data.get("content", "")).strip()
            if not text:
                text = str(data.get("thinking", "")).strip()
        latency_ms_total = (time.time() - t0) * 1000
        meta: Dict[str, Any] = {
            "latency_ms_total": latency_ms_total,
            "latency_ms_retrieve": 0.0,
            "latency_ms_generate": latency_ms_total,
            "response_keys": response_keys,
            "had_empty_response": not bool(text),
            "streaming": stream,
        }
        if stream:
            meta["chunk_count"] = chunk_count
        return text, meta
