from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

class RagApiClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ensure_index(self, data_sources: List[str], cache_dir: Optional[str] = None, force_rebuild: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"data_sources": data_sources, "force_rebuild": force_rebuild}
        if cache_dir is not None:
            payload["cache_dir"] = cache_dir
        r = requests.post(f"{self.base_url}/index", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def answer(self, question: str, top_n: Optional[int], score_threshold: Optional[float]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"question": question}
        if top_n is not None:
            payload["top_n"] = top_n
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        r = requests.post(f"{self.base_url}/answer", json=payload, timeout=self.timeout)
        if r.status_code == 409:
            data = r.json()
            raise RuntimeError(data.get("error", "RAG_NOT_READY"))
        r.raise_for_status()
        data = r.json()
        if "meta" in data and isinstance(data["meta"], dict):
            data["meta"].setdefault("retrieval_count", len(data.get("contexts", []) or []))
        return {"answer": data.get("answer",""), "contexts": data.get("contexts",[]), "meta": data.get("meta",{})}
