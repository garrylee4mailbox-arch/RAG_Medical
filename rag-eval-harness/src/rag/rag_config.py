from __future__ import annotations

import os
from pathlib import Path
from typing import List

# -----------------------------
# Defaults (overridable by env)
# -----------------------------
DEFAULT_TOP_N: int = int(os.getenv("RAG_TOP_N", "5"))
DEFAULT_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))

# ✅ 你的真实数据根目录（建议用环境变量覆盖；否则用你本机路径）
DEFAULT_DATA_SOURCES: List[str] = (
    os.getenv("RAG_DATA_SOURCES").split(os.pathsep)
    if os.getenv("RAG_DATA_SOURCES")
    else [r"C:\Users\GuanlinLi\Desktop\RAG_Medical\Chinese-medical-dialogue-data-master"]
)

# ✅ 你的“最优解”RAG本体文件
DEFAULT_USER_RAG_PATH: str = os.getenv("USER_RAG_PATH", "Original_Version3.0.py")


def resolve_user_rag_path() -> Path:
    """
    Resolve USER_RAG_PATH to an absolute path.
    - If relative, assume it is relative to repo root.
    """
    p = Path(DEFAULT_USER_RAG_PATH)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]  # src/rag/rag_config.py -> repo root
        p = repo_root / p
    return p.resolve()