from __future__ import annotations

# 这是“OpenAI兼容接口”的通用客户端。
# 现在可用于阿里云百炼 / DashScope。
# 以后也可切换到其他兼容 OpenAI 接口的平台。
# 如果要切换平台，优先改配置和环境变量，不要先改代码。

import time
from typing import Any, Dict, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package not installed")
        if not api_key:
            raise ValueError("Compatible API key not set")
        if not base_url:
            raise ValueError("Compatible API base_url not set")
        if not model:
            raise ValueError("Compatible API model not set")

        # 兼容接口平台通常直接复用 OpenAI SDK；后续换平台时优先改配置。
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens

        resp = self.client.chat.completions.create(
            **request,
        )
        latency_ms_total = (time.time() - t0) * 1000
        text = resp.choices[0].message.content or ""
        meta: Dict[str, Any] = {
            "latency_ms_total": latency_ms_total,
            "latency_ms_retrieve": 0.0,
            "latency_ms_generate": latency_ms_total,
        }
        return text.strip(), meta
