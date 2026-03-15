from __future__ import annotations

"""
Adapter around user RAG file (Original_Version2.0.py) without interactive loops.

Required exported interface (preferred):
1) prepare_rag(data_sources: list[str], cache_dir: str|None) -> None
2) rag_answer(question: str, *, top_n: int, score_threshold: float) -> dict

Return dict schema:
- answer: str
- contexts: list[{rank, department, source, score, text}]
- meta: dict (must include retrieval_count; may include used_global_pool, etc.)
"""

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .rag_config import resolve_user_rag_path, DEFAULT_TOP_N, DEFAULT_SCORE_THRESHOLD

_user_module: Optional[types.ModuleType] = None
_prepare_fn: Optional[Callable[..., Any]] = None
_answer_fn: Optional[Callable[..., Any]] = None

def _load_user_module() -> types.ModuleType:
    global _user_module
    if _user_module is not None:
        return _user_module
    path: Path = resolve_user_rag_path()
    if not path.exists():
        raise FileNotFoundError(
            f"USER_RAG_PATH not found: {path}. Put Original_Version2.0.py in repo root or set USER_RAG_PATH."
        )
    spec = importlib.util.spec_from_file_location("user_rag", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import user RAG from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore
    _user_module = module
    return module

def _resolve_functions(module: types.ModuleType) -> None:
    global _prepare_fn, _answer_fn
    if _prepare_fn is not None and _answer_fn is not None:
        return

    prepare_candidates = ["prepare_rag", "build_index", "prepare"]
    answer_candidates = ["rag_answer", "answer", "generate_answer", "chat"]

    for n in prepare_candidates:
        fn = getattr(module, n, None)
        if callable(fn):
            _prepare_fn = fn
            break

    for n in answer_candidates:
        fn = getattr(module, n, None)
        if callable(fn):
            _answer_fn = fn
            break

    if _prepare_fn is None:
        raise NotImplementedError(
            "User RAG missing prepare function. Please implement:\n"
            "  prepare_rag(data_sources: list[str], cache_dir: str|None) -> None\n"
            "or build_index(data_sources, cache_dir) -> None"
        )
    if _answer_fn is None:
        raise NotImplementedError(
            "User RAG missing answer function. Please implement:\n"
            "  rag_answer(question: str, *, top_n: int, score_threshold: float) -> dict\n"
            "or answer/generate_answer with similar signature."
        )

def prepare_rag(data_sources: List[str], cache_dir: Optional[str] = None) -> None:
    m = _load_user_module()
    _resolve_functions(m)
    assert _prepare_fn is not None
    sig = inspect.signature(_prepare_fn)
    kwargs: Dict[str, Any] = {}
    if "data_sources" in sig.parameters:
        kwargs["data_sources"] = data_sources
    else:
        # best-effort positional
        kwargs[list(sig.parameters.keys())[0]] = data_sources
    if "cache_dir" in sig.parameters:
        kwargs["cache_dir"] = cache_dir
    elif len(sig.parameters) >= 2:
        kwargs[list(sig.parameters.keys())[1]] = cache_dir
    _prepare_fn(**kwargs)

def rag_answer(question: str, *, top_n: int = DEFAULT_TOP_N, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> Dict[str, Any]:
    m = _load_user_module()
    _resolve_functions(m)
    assert _answer_fn is not None

    try:
        result = _answer_fn(question, top_n=top_n, score_threshold=score_threshold)
    except TypeError:
        result = _answer_fn(question)

    if not isinstance(result, dict):
        raise ValueError("User RAG answer function must return dict")

    answer_text = str(result.get("answer", result.get("response", ""))).strip()

    raw_contexts = result.get("contexts") or result.get("sources") or []
    contexts: List[Dict[str, Any]] = []
    if isinstance(raw_contexts, dict):
        raw_contexts = list(raw_contexts.values())
    if isinstance(raw_contexts, list):
        for i, ctx in enumerate(raw_contexts):
            if isinstance(ctx, dict):
                contexts.append({
                    "rank": int(ctx.get("rank", i+1)),
                    "department": str(ctx.get("department", ctx.get("dept", ""))),
                    "source": str(ctx.get("source", ctx.get("doc_id", ""))),
                    "score": float(ctx.get("score", ctx.get("similarity", 0.0))),
                    "text": str(ctx.get("text", ctx.get("content", ""))),
                })
            else:
                contexts.append({
                    "rank": i+1, "department": "", "source": "", "score": 0.0, "text": str(ctx)
                })

    meta: Dict[str, Any] = {"retrieval_count": len(contexts)}
    if isinstance(result.get("meta"), dict):
        meta.update(result["meta"])

    return {"answer": answer_text, "contexts": contexts, "meta": meta}
