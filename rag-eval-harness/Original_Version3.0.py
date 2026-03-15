"""
================================================================================
 Original_Version2.0_REFACTORED.py

【中文说明（先中文后英文）】
--------------------------------------------------------------------------------
这是对你当前 Original_Version2.0.py 的“最优解改造版”：
1) 不改变你的核心算法思路：加载CSV → embedding → VECTOR_DB → retrieve → LLM生成
2) 新增两个稳定入口函数（科研级接口）：
   - prepare_rag(data_sources, cache_dir=None)
   - rag_answer(question, top_n=..., score_threshold=...)
   这两者是给：
   - eval harness（脚手架）
   - FastAPI 服务
   - 单元测试 / ablation / 参数扫描
   使用的统一入口。
3) 保留脚本运行方式：仍可直接 python 本文件进入交互聊天（兼容旧习惯）
4) 关键工程原则：import 不触发重建索引、不触发 input 循环
   - 重建索引只发生在 prepare_rag() / main() 里
   - 这样评测、服务、论文复现实验更稳定可控

【English Notes】
--------------------------------------------------------------------------------
This is a “best-practice refactor” of your Original_Version2.0.py:
1) Keeps your core pipeline intact: CSV loading → embedding → VECTOR_DB → retrieve → LLM generation
2) Adds two stable, research-grade public APIs:
   - prepare_rag(data_sources, cache_dir=None)
   - rag_answer(question, top_n=..., score_threshold=...)
   These are the unified entry points for:
   - evaluation harness
   - FastAPI service
   - tests / ablation / parameter sweeps
3) Preserves script-style usage: you can still run `python this_file.py` to start an interactive chatbot
4) Key engineering rule: importing this module must not trigger heavy computation or interactive loops
   - Index building happens only in prepare_rag() / main()

⚠️ Medical Disclaimer / 医疗免责声明
--------------------------------------------------------------------------------
本项目仅用于研究与教育目的，不构成医疗诊断或治疗建议。
This project is for research/education only and is not medical advice.
================================================================================
"""

from __future__ import annotations

import os
import csv
import time
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# 【中文】可选依赖：ollama Python 包（你原代码使用 ollama.embeddings / ollama.chat）
#        如果没有安装，会给出明确错误提示。
# 【English】Optional dependency: ollama Python package. If missing, raise a clear error.
# -----------------------------------------------------------------------------
try:
    import ollama  # type: ignore
except Exception as e:  # noqa
    ollama = None
    _OLLAMA_IMPORT_ERROR = e


# =============================================================================
# 1) CONFIG / 配置区
# =============================================================================

# 【中文】Embedding 模型与 LLM 模型名（保持你原设置）
# 【English】Embedding + LLM model names (kept as in your original code)
EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = "llama3"

# 【中文】全局向量库：存储每条 chunk 的 embedding + metadata
# 【English】Global vector DB: each entry stores embedding + metadata
VECTOR_DB: List[Dict[str, Any]] = []

# 【中文】默认数据目录（建议未来用环境变量覆盖）
# 【English】Default dataset directory (can be overridden by environment variable)
DATA_DIR = os.getenv(
    "MEDICAL_RAG_DATA_DIR",
    r"D:\RAG development\RAG_Medical\Chinese-medical-dialogue-data-master",
)

# 【中文】索引状态与构建信息：用于服务/评测可追溯
# 【English】Index state & build info for traceability
_RAG_READY: bool = False
_INDEX_INFO: Dict[str, Any] = {}


# =============================================================================
# 2) Department Mapping / 科室映射（保持你当前版本）
# =============================================================================

# 【中文】儿科子科室集合
# 【English】Pediatric sub-departments set
PEDIATRIC_SUBDEPTS = {
    "耳鼻喉科",
    "神经内科",
    "新生儿科",
    "营养保健科",
}

# 【中文】肿瘤科子目录集合
# 【English】Tumor-related directory names set
TUMOR_DIR_NAMES = {
    "膀胱癌", "鼻咽癌", "大肠癌", "胆管癌", "肺癌", "肝癌", "宫颈癌", "骨癌",
    "黑色素瘤", "甲状腺癌", "甲状腺腺癌", "结肠癌", "口腔癌", "口腔颌面肿瘤",
    "淋巴瘤", "卵巢癌", "脑瘤", "前列腺癌", "乳腺癌", "乳腺纤维腺癌", "软纤维瘤",
    "肾癌", "外阴癌", "胃癌", "小儿血管瘤", "小儿肿瘤", "血管瘤", "胰腺癌",
    "直肠癌", "肿瘤疾病", "肿瘤与疼痛", "子宫癌",
}

# 【中文】标准科室映射（顶层）
# 【English】Canonical department mapping (top-level)
CANONICAL_DEPT = {
    "儿科": {"儿科"} | PEDIATRIC_SUBDEPTS,
    "肿瘤科": {"肿瘤科"} | TUMOR_DIR_NAMES,
}


def infer_department_from_filename(filepath: str) -> str:
    """
    【中文】
    根据文件路径推断所属科室（不依赖 CSV 内字段）。
    规则（保持你原实现）：
    1) 父目录名在 CANONICAL_DEPT 的 alias 集合中 → 返回对应科室
    2) 父目录名在 TUMOR_DIR_NAMES → 肿瘤科
    3) 父目录名在 PEDIATRIC_SUBDEPTS → 儿科
    4) 否则 → 未知科室

    【English】
    Infer department from file path (no CSV dependency).
    Same rules as your original implementation.
    """
    folder_name = os.path.basename(os.path.dirname(filepath))

    for dept, aliases in CANONICAL_DEPT.items():
        if folder_name in aliases:
            return dept

    if folder_name in TUMOR_DIR_NAMES:
        return "肿瘤科"

    if folder_name in PEDIATRIC_SUBDEPTS:
        return "儿科"

    return "未知科室"


def cancer_rule(query_lower: str) -> set[str]:
    """
    【中文】癌症/肿瘤关键词触发规则（软引导）。
    【English】Cancer-related keyword triggers (soft guidance).
    """
    cancer_triggers = ["癌", "肿瘤", "恶性", "化疗", "放疗", "靶向", "免疫治疗", "肿瘤标志物"]
    if any(t in query_lower for t in cancer_triggers):
        return {"肿瘤科", "耳鼻喉科"}
    return set()


def normalize_dept(raw: str) -> str:
    """
    【中文】
    将原始科室名映射成标准科室名（静态映射）。
    用途：构建向量库时统一 metadata['department']。

    【English】
    Map raw department name to canonical department name.
    Used to normalize metadata['department'] during indexing.
    """
    for canon, aliases in CANONICAL_DEPT.items():
        if raw in aliases:
            return canon
    return raw


def infer_departments(query: str) -> set[str]:
    """
    【中文】
    基于用户问题做“软科室推断”，用于检索阶段的优先过滤。
    若未匹配到关键词，则默认返回 {'儿科', '肿瘤科'}（不加限制）。

    【English】
    Infer possible relevant departments from the query for soft filtering.
    If no keywords matched, return {'儿科','肿瘤科'} as no restriction.
    """
    q = query.lower()
    inferred = set()
    if any(k in q for k in ["癌", "肿瘤", "放疗", "化疗"]):
        inferred.add("肿瘤科")
    if any(k in q for k in ["小儿", "宝宝", "儿童", "婴儿"]):
        inferred.add("儿科")
    if not inferred:
        inferred = {"儿科", "肿瘤科"}
    return inferred


# =============================================================================
# 3) Data Loading / 数据加载
# =============================================================================

def load_medical_chunks(data_folder: str) -> Tuple[List[str], List[str], List[str]]:
    """
    【中文】
    遍历数据目录，读取各子目录的 CSV，组装成三元组并返回：
      - chunks:       "[Title]...[Patient]...[Doctor]..." 形式
      - sources:      每条 chunk 对应 CSV 文件路径
      - departments:  每条 chunk 的科室（优先 CSV 'department'，否则用子目录名）

    【English】
    Scan the dataset folder, read CSV files, and return (chunks, sources, departments).
    """
    chunks: List[str] = []
    sources: List[str] = []
    departments: List[str] = []

    encodings_to_try = ["gb18030", "gbk", "utf-8", "gb2312"]

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            if not file.endswith(".csv"):
                continue

            filepath = os.path.join(subdir_path, file)
            print(f"📂 Reading file: {filepath}")

            success = False
            for enc in encodings_to_try:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        reader = csv.DictReader(f)
                        count = 0
                        for row in reader:
                            q = (row.get("ask") or "").strip()
                            a = (row.get("answer") or "").strip()
                            title = (row.get("title") or "").strip()

                            dept = (row.get("department") or "").strip() or os.path.basename(subdir_path)

                            if q and a:
                                chunk = f"[Title]: {title}\n[Patient]: {q}\n[Doctor]: {a}"
                                chunks.append(chunk)
                                sources.append(filepath)
                                departments.append(dept)
                                count += 1

                        print(f"✅ Loaded {count} Q&A pairs with encoding: {enc}")
                        success = True
                        break

                except Exception as e:
                    print(f"❌ Failed with encoding {enc}: {e}")

            if not success:
                print(f"❌ Could not read file: {filepath} with any known encoding.")

    print(f"\n✅ Total loaded QA chunks: {len(chunks)}")
    return chunks, sources, departments


# =============================================================================
# 4) Embeddings / 嵌入
# =============================================================================

def _require_ollama() -> None:
    """
    【中文】确保 ollama 包可用，否则给出明确安装提示。
    【English】Ensure ollama package is available; otherwise raise with install hint.
    """
    if ollama is None:
        raise RuntimeError(
            "Python package 'ollama' is not available. "
            "Please run: pip install ollama\n"
            f"Original import error: {_OLLAMA_IMPORT_ERROR}"
        )


def get_text_embedding(text: str, max_length: int = 800) -> List[float]:
    """
    【中文】
    调用 Ollama embeddings 获取文本向量。
    - 超长文本会被截断到 max_length（默认 800）
    - 最多 3 次指数退避重试

    【English】
    Call Ollama embeddings to get text vectors.
    - Truncate long text to max_length
    - Up to 3 retries with exponential backoff
    """
    _require_ollama()

    if len(text) > max_length:
        text = text[:max_length]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return resp["embedding"]
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            sleep(2 ** attempt)


# =============================================================================
# 5) Vector DB Build / 向量库构建
# =============================================================================

def build_vector_db(chunks: List[str], sources: List[str], departments: List[str]) -> None:
    """
    【中文】
    对每条问答做嵌入，写入全局 VECTOR_DB。
    metadata:
      - source: CSV 路径
      - department: 标准化后的科室名

    【English】
    Embed each chunk and write into global VECTOR_DB with metadata.
    """
    VECTOR_DB.clear()

    for i, (chunk, source, dept) in enumerate(zip(chunks, sources, departments), 1):
        try:
            embedding = get_text_embedding(chunk)
            VECTOR_DB.append(
                {
                    "id": i,
                    "text": chunk,
                    "embedding": np.array(embedding, dtype=np.float32),
                    "metadata": {
                        "source": source,
                        "department": normalize_dept(dept),
                    },
                }
            )
            if i % 200 == 0:
                print(f"📌 Embedded {i}/{len(chunks)} ...")
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    【中文】余弦相似度。
    【English】Cosine similarity.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# 6) Retrieval / 检索
# =============================================================================

def retrieve(query: str, top_n: int = 5, score_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    【中文】
    向量检索 + 软科室过滤/加权（保持你当前版本的逻辑特点）：
    1) infer_departments(query) 推断可能科室
    2) query 进行关键词增强 & hint 加前缀
    3) pool 软过滤：若过滤后太少则回退全库
    4) similarity + bonus 分数
    5) 若无结果则降低阈值再试一次
    6) 返回 top_n

    【English】
    Vector retrieval with soft department filtering & score bonus, including retry with lower threshold.
    """
    if not VECTOR_DB:
        return []

    # 1) infer departments
    inferred = infer_departments(query)

    try:
        # 2) keyword enhancement (same spirit as your version)
        DEPT_KEY_HINTS = {
            "肿瘤科": ["癌", "肿瘤", "放疗", "化疗", "术后", "恶性"],
            "儿科": ["儿童", "小孩", "宝宝", "中耳炎", "发烧"],
        }
        keywords: List[str] = []
        for dept, hints in DEPT_KEY_HINTS.items():
            if any(h in query for h in hints):
                keywords.extend(hints[:2])
        if keywords:
            query = f"{query} {' '.join(keywords)}"

        query_with_hint = f"【可能科室：{', '.join(inferred)}】{query}"
        query_vector = np.array(get_text_embedding(query_with_hint), dtype=np.float32)

    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        return []

    # 3) pool selection
    if inferred:
        pool = [it for it in VECTOR_DB if it["metadata"].get("department") in inferred]
    else:
        pool = VECTOR_DB

    if len(pool) < 0.1 * len(VECTOR_DB):
        pool = VECTOR_DB

    # 4) score
    def _score_items(th: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in pool:
            base = cosine_similarity(query_vector, item["embedding"])
            if base < th:
                continue
            dept = item["metadata"].get("department")
            dept = normalize_dept(dept) if dept else dept
            bonus = 0.05 if (inferred and dept in inferred) else 0.0
            score = base + bonus

            result = dict(item)
            result["score"] = score
            result["base_score"] = base
            out.append(result)
        return out

    results = _score_items(score_threshold)

    # 5) retry if empty
    if not results:
        new_th = score_threshold - 0.1
        print(f"⚠️ No results above {score_threshold:.2f}, retrying with threshold {new_th:.2f}...")
        results = _score_items(new_th)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# =============================================================================
# 7) Generation / 生成
# =============================================================================

def _generate_answer_text(user_input: str, hits: List[Dict[str, Any]], max_ctx: int = 2) -> str:
    """
    【中文】
    “科研/评测用”的回答生成：
    - 返回纯回答文本（不强行把 Retrieved context 拼到 answer 里）
    - contexts 会由 rag_answer() 单独返回结构化列表，方便评测和分析

    【English】
    Generation for evaluation/service:
    - Return answer text only (contexts are returned separately by rag_answer()).
    """
    _require_ollama()

    if not hits:
        return "⚠️ No relevant information found."

    # truncate contexts
    hits = hits[:max_ctx]

    context_lines: List[str] = []
    for i, item in enumerate(hits, 1):
        dept = item["metadata"].get("department", "未知科室")
        src = os.path.basename(item["metadata"].get("source", ""))
        score = f"{item.get('base_score', item.get('score', 0.0)):.2f}"
        text_preview = item["text"].strip()
        context_lines.append(f"[{i}] ({dept}, score={score}, src={src})\n{text_preview}")
    context_block = "\n\n".join(context_lines)

    system_prompt = f"""
你是一名谨慎的医学助理，请根据以下检索到的医疗问答内容，使用中英双语回答问题。

格式要求：
1) 输出两部分：
   🧠 中文回答部分：
   💡 English summary:
2) 用正式、简洁、专业的语气；
3) 不要编造或臆测；
4) 若信息不足，请指出不确定性，并建议就医或咨询专业医生。

---------------------
用户提问：
{user_input}

参考内容：
{context_block}
---------------------
""".strip()

    resp = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        stream=False,
    )
    return resp["message"]["content"].strip()


def compose_answer(user_input: str, hits: List[Dict[str, Any]]) -> str:
    """
    【中文】
    “交互聊天显示用”的回答生成（保留你当前体验）：
    - 会把 Retrieved context 和最终回答一起拼出来，便于你人工调试观察

    【English】
    “Interactive debugging” output:
    - Includes retrieved context + final answer for readability during manual runs.
    """
    if len(hits) > 2:
        print(f"ℹ️ Retrieved {len(hits)} results, truncating to top 2 for concise answer.")
        hits = hits[:2]

    context_lines = []
    for i, item in enumerate(hits, 1):
        dept = item["metadata"].get("department", "未知科室")
        src = os.path.basename(item["metadata"].get("source", ""))
        score = f"{item.get('base_score', item.get('score', 0.0)):.2f}"
        text_preview = item["text"].strip().replace("\n", " ")
        context_lines.append(f"[{i}] ({dept}, score={score}, src={src})\n{text_preview}")
    context_block = "\n\n".join(context_lines)

    try:
        answer_text = _generate_answer_text(user_input, hits, max_ctx=2)
    except Exception as e:
        return f"⚠️ 生成失败：{e}"

    # Split bilingual parts if present (keeps your structure)
    zh_part = ""
    en_part = ""
    if "💡 English summary" in answer_text:
        zh_part, en_part = answer_text.split("💡 English summary", 1)
        zh_part = zh_part.replace("🧠 中文回答部分：", "").strip()
        en_part = en_part.lstrip("\n").strip()
    else:
        zh_part = answer_text.strip()
        en_part = "(No English summary generated)"

    final_output = (
        f"📚 Retrieved context:\n"
        f"{context_block}\n\n"
        f"──────────────────────────────\n"
        f"🧠 中文回答部分：\n{zh_part}\n\n"
        f"──────────────────────────────\n"
        f"💡 English summary{en_part}"
    )
    return final_output


# =============================================================================
# 8) ✅ Research-grade Stable APIs / 科研级稳定接口（最优解核心）
# =============================================================================

def _resolve_data_dir_from_sources(data_sources: List[str]) -> str:
    """
    【中文】
    将 harness/service 传入的 data_sources 解析成数据根目录：
    - 如果传入的是目录：直接用该目录
    - 如果传入的是文件：尽量用其父目录或父父目录
    - 否则回退到 DATA_DIR

    【English】
    Resolve data_sources into a dataset root directory.
    """
    if not data_sources:
        return DATA_DIR

    for p in data_sources:
        ap = os.path.abspath(p)
        if os.path.isdir(ap):
            return ap

    for p in data_sources:
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            parent = os.path.dirname(ap)
            parent2 = os.path.dirname(parent)
            if os.path.isdir(parent2):
                return parent2
            if os.path.isdir(parent):
                return parent

    return DATA_DIR


def prepare_rag(data_sources: List[str], cache_dir: Optional[str] = None) -> None:
    """
    【中文】
    ✅ 必须实现（给评测框架/服务使用）
    构建/加载索引（你的实现即：加载CSV → build_vector_db 写入 VECTOR_DB）。
    - 不要在 import 时自动执行
    - 只在此函数内执行重计算，保证可控复现

    参数：
    - data_sources: 数据根目录或若干路径（目录/文件均可）
    - cache_dir: 预留（你未来可把 embedding / db 缓存到此目录）

    【English】
    Required API for harness/service.
    Build/load index under controlled execution (no side effects on import).
    """
    global _RAG_READY, _INDEX_INFO, DATA_DIR

    t0 = time.time()

    # Resolve dataset root
    resolved_dir = _resolve_data_dir_from_sources(data_sources)
    DATA_DIR = resolved_dir  # allow override

    # Load dataset
    chunks, sources, departments = load_medical_chunks(data_folder=DATA_DIR)

    if not chunks:
        _RAG_READY = False
        raise RuntimeError(f"No data loaded from: {DATA_DIR}")

    # Stable de-duplication (keeps order)
    seen = set()
    unique_data = []
    for triplet in zip(chunks, sources, departments):
        if triplet not in seen:
            unique_data.append(triplet)
            seen.add(triplet)

    chunks_u, sources_u, departments_u = zip(*unique_data)

    # Build vector db (embeddings)
    build_vector_db(list(chunks_u), list(sources_u), list(departments_u))

    t1 = time.time()

    _RAG_READY = True
    _INDEX_INFO = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": data_sources,
        "resolved_data_dir": resolved_dir,
        "cache_dir": cache_dir,
        "count": len(VECTOR_DB),
        "build_seconds": round(t1 - t0, 3),
        "embedding_model": EMBEDDING_MODEL,
        "language_model": LANGUAGE_MODEL,
    }


def rag_answer(question: str, *, top_n: int = 5, score_threshold: float = 0.6) -> Dict[str, Any]:
    """
    【中文】
    ✅ 必须实现（给评测框架/服务使用）
    返回结构必须稳定，便于评测与审计：

    {
      "answer": str,
      "contexts": [
        {"rank": int, "department": str, "source": str, "score": float, "text": str}
      ],
      "meta": {
        "retrieval_count": int,
        "latency_ms_total": int,
        "latency_ms_retrieve": int,
        "latency_ms_generate": int,
        "index_info": {...}
      }
    }

    【English】
    Required API for harness/service with stable response schema.
    """
    if not _RAG_READY:
        raise RuntimeError("RAG_NOT_READY: call prepare_rag() first")

    t0 = time.time()
    hits = retrieve(question, top_n=top_n, score_threshold=score_threshold)
    t1 = time.time()

    answer_text = _generate_answer_text(question, hits, max_ctx=2) if hits else "⚠️ No relevant information found."
    t2 = time.time()

    contexts: List[Dict[str, Any]] = []
    for i, item in enumerate(hits, 1):
        meta = item.get("metadata", {}) or {}
        contexts.append(
            {
                "rank": i,
                "department": str(meta.get("department", "未知科室")),
                "source": str(meta.get("source", "")),
                "score": float(item.get("score", item.get("base_score", 0.0))),
                "text": str(item.get("text", "")),
            }
        )

    return {
        "answer": answer_text,
        "contexts": contexts,
        "meta": {
            "retrieval_count": len(contexts),
            "latency_ms_retrieve": int((t1 - t0) * 1000),
            "latency_ms_generate": int((t2 - t1) * 1000),
            "latency_ms_total": int((t2 - t0) * 1000),
            "index_info": dict(_INDEX_INFO),
        },
    }


# =============================================================================
# 9) Interactive Chatbot / 交互主循环（保留旧体验）
# =============================================================================

def run_chatbot() -> None:
    """
    【中文】
    交互主循环：保持你原来的体验
    - 输入问题
    - 检索 top chunks
    - compose_answer 输出（含 context + 双语回答）

    【English】
    Interactive chatbot loop for manual debugging / demos.
    """
    print("\n🤖 Welcome to the RAG Chatbot! Type your question or 'exit' to quit.\n")

    while True:
        user_input = input("Ask me a question: ")
        if user_input.strip().lower() == "exit":
            print("👋 Goodbye!")
            break

        top_chunks = retrieve(user_input, top_n=5, score_threshold=0.6)
        if not top_chunks:
            print("⚠️ No relevant information found.\n")
            continue

        print("\n💬 Chatbot response:")
        fused = compose_answer(user_input, top_chunks)
        print(fused)


# =============================================================================
# 10) Script Entry / 脚本入口（兼容旧运行方式）
# =============================================================================

def main() -> None:
    """
    【中文】
    脚本入口：加载数据 → 构建向量库 → 进入交互聊天
    注意：这里调用 prepare_rag()，使逻辑与科研接口一致（统一入口）。

    【English】
    Script entry: build index then start interactive loop.
    Uses prepare_rag() to keep a single source of truth.
    """
    print("📥 Loading data and building vector database...")

    # 这里把 DATA_DIR 作为 data_sources 传入，保持旧用法兼容
    prepare_rag(data_sources=[DATA_DIR], cache_dir=None)

    print("✅ Vector DB ready.")
    run_chatbot()


if __name__ == "__main__":
    main()