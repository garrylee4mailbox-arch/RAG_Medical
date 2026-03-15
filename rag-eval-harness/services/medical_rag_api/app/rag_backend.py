from __future__ import annotations

import hashlib, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ensure repo root on sys.path, so `import src...` works when running server.py directly
_this = Path(__file__).resolve()
_repo_root = _this.parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.rag import rag_adapter  # type: ignore

def _fingerprint(data_sources: List[str], defaults: Dict[str, Any]) -> str:
    sha = hashlib.sha256()
    sha.update(str(sorted(data_sources)).encode("utf-8"))
    sha.update(str(defaults).encode("utf-8"))
    for src in sorted(data_sources):
        p = Path(src)
        if not p.is_absolute():
            p = _repo_root / src
        if not p.exists():
            sha.update(f"MISSING:{p}".encode("utf-8"))
            continue
        if p.is_file():
            sha.update(str(p.resolve()).encode("utf-8"))
            sha.update(str(int(p.stat().st_mtime)).encode("utf-8"))
        else:
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    fp = Path(root)/fn
                    sha.update(str(fp.resolve()).encode("utf-8"))
                    sha.update(str(int(fp.stat().st_mtime)).encode("utf-8"))
    return sha.hexdigest()

class RagBackend:
    def __init__(self) -> None:
        self.rag_ready = False
        self.index_info: Dict[str, Any] = {}

    def build_index(self, data_sources: List[str], cache_dir: Optional[str], force_rebuild: bool) -> Dict[str, Any]:
        defaults = {"top_n": None, "score_threshold": None}
        fp = _fingerprint(data_sources, defaults)
        if self.rag_ready and (not force_rebuild) and self.index_info.get("fingerprint") == fp:
            return {"rag_ready": True, "index_info": self.index_info}

        rag_adapter.prepare_rag(data_sources=data_sources, cache_dir=cache_dir)
        self.index_info = {
            "data_sources": data_sources,
            "built_at": datetime.utcnow().isoformat() + "Z",
            "count": None,
            "fingerprint": fp,
        }
        self.rag_ready = True
        return {"rag_ready": True, "index_info": self.index_info}

    def answer(self, question: str, top_n: Optional[int], score_threshold: Optional[float]) -> Dict[str, Any]:
        if not self.rag_ready:
            raise RuntimeError("RAG_NOT_READY")
        t0 = time.time()
        # we can't split retrieve/generate cleanly without user RAG exposing it; best-effort placeholders
        res = rag_adapter.rag_answer(question, top_n=top_n or 5, score_threshold=score_threshold or 0.0)
        total_ms = (time.time() - t0) * 1000
        meta = dict(res.get("meta", {}) if isinstance(res.get("meta"), dict) else {})
        meta.setdefault("retrieval_count", len(res.get("contexts", []) or []))
        meta["latency_ms_total"] = total_ms
        meta.setdefault("latency_ms_retrieve", 0.0)
        meta.setdefault("latency_ms_generate", total_ms)
        meta["fingerprint"] = self.index_info.get("fingerprint")
        return {"answer": res.get("answer",""), "contexts": res.get("contexts",[]), "meta": meta}
