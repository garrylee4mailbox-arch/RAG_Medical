import numpy as np
import ollama
from time import sleep
import os
import pandas as pd
import csv

"""
================================================================================
 LATEST_dept_mapping_readable.py

 ✨ 说明（仅做可读性优化，不改动逻辑）：
 - 保留原有功能与实现，不新增/删除任何业务逻辑。
 - 统一分区：配置、科室映射、数据加载、嵌入、向量库、检索、生成、主循环。
 - 为每个函数添加了较详细的 docstring 与行内注释，便于维护与调试。
 - 标注了两个当前代码中被调用但未定义的函数（normalize_dept、infer_departments）。
   这些仅以注释的形式提醒，未作实现改动。
================================================================================
"""

# === CONFIGURATION ===
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3'
TEXT_FILE = 'cat-facts.txt'
VECTOR_DB = []  # Will hold all embeddings and text chunks
DATA_DIR = r"D:\RAG development\RAG_Medical\Chinese-medical-dialogue-data-master" #absolute path to the data directory
# ==============================================================================

# ==============================================================================
#  科室映射：儿科与肿瘤科（来自你的最新要求）
# ============================================================================
# 儿科的子科室集合
PEDIATRIC_SUBDEPTS = {
    "耳鼻喉科",
    "神经内科",
    "新生儿科",
    "营养保健科",
}

# 肿瘤科的子目录集合（癌症/肿瘤类疾病）
TUMOR_DIR_NAMES = {
    "膀胱癌", "鼻咽癌", "大肠癌", "胆管癌", "肺癌", "肝癌", "宫颈癌", "骨癌",
    "黑色素瘤", "甲状腺癌", "甲状腺腺癌", "结肠癌", "口腔癌", "口腔颌面肿瘤",
    "淋巴瘤", "卵巢癌", "脑瘤", "前列腺癌", "乳腺癌", "乳腺纤维腺癌", "软纤维瘤",
    "肾癌", "外阴癌", "胃癌", "小儿血管瘤", "小儿肿瘤", "血管瘤", "胰腺癌",
    "直肠癌", "肿瘤疾病", "肿瘤与疼痛", "子宫癌",
}

# 标准科室映射（顶层）
CANONICAL_DEPT = {
    "儿科": {"儿科"} | PEDIATRIC_SUBDEPTS,
    "肿瘤科": {"肿瘤科"} | TUMOR_DIR_NAMES,
}


# ==============================================================================
#  路径 → 科室推断 & 癌症规则（未更改逻辑，仅加注释）
# ==============================================================================
def infer_department_from_filename(filepath: str) -> str:
    """根据文件路径推断所属科室（不依赖 CSV 内字段）。

    判定顺序：
    1) 父级目录名若在 CANONICAL_DEPT 的别名集合中 → 返回对应科室；
        如果文件所在的文件夹名字在“标准科室映射”里能找到匹配的别名
       （例如目录名=“宫颈癌”对应肿瘤科），那就直接认定这个文件属于那个科室。
    2) 目录名若是肿瘤类目录（TUMOR_DIR_NAMES） → 归入『肿瘤科』；
    3) 目录名若在儿科子科室（PEDIATRIC_SUBDEPTS） → 归入『儿科』；
    4) 否则 → 返回『未知科室』。

    注：保持原实现，不改动任何判断条件。
    """
    folder_name = os.path.basename(os.path.dirname(filepath))

    # 先看是否正好就是某个标准科室/别名
    for dept, aliases in CANONICAL_DEPT.items():
        if folder_name in aliases:
            return dept

    # 再看是否属于肿瘤类目录
    if folder_name in TUMOR_DIR_NAMES:
        return "肿瘤科"

    # 再看是否属于儿科下的子科室
    if folder_name in PEDIATRIC_SUBDEPTS:
        return "儿科"

    return "未知科室"

def cancer_rule(query_lower: str) -> set[str]:
    """基于关键词的『癌症/肿瘤』触发规则（用于软引导检索科室）。

    规则：
    - 若用户查询包含『癌/肿瘤/恶性/化疗/放疗/靶向/免疫治疗/肿瘤标志物』等词，
      则返回 {'肿瘤科', '耳鼻喉科'}，其中耳鼻喉科用于覆盖鼻咽癌等头颈部肿瘤。

    注意：只是提供一个可能的部门提示集合，本文件未改动其调用位置/方式。
    """
    cancer_triggers = ["癌", "肿瘤", "恶性", "化疗", "放疗", "靶向", "免疫治疗", "肿瘤标志物"]
    if any(t in query_lower for t in cancer_triggers):
        return {"肿瘤科", "耳鼻喉科"}
    return set()

def normalize_dept(raw: str) -> str:
    """将原始科室名规范化，若未识别则返回原名"""
    """
    功能：将原始科室名（来自文件夹或 CSV）映射成标准科室名。
    用途：在构建向量数据库阶段，统一不同目录或别名的科室标识。
    举例：
        "宫颈癌" → "肿瘤科"
        "耳鼻喉科" → "儿科"
    特点：静态映射，不依赖用户输入。
    """
    for canon, aliases in CANONICAL_DEPT.items():
        if raw in aliases:
            return canon
    return raw
    
def infer_departments(query: str) -> set[str]:
    """基于关键词的简单科室推断"""
    """
    功能：根据用户提问内容动态推断可能相关的科室。
    用途：在检索阶段做“软过滤”，优先在匹配科室的样本中查找答案。
    举例：
        "做过宫颈癌手术后上半身刺痛怎么办" → {"肿瘤科"}
        "三岁小孩中耳炎流脓水" → {"儿科"}
    特点：动态判断，面向用户查询而非数据本身。
    """
    q = query.lower()
    inferred = set()
    if any(k in q for k in ["癌", "肿瘤", "放疗", "化疗"]):
        inferred.add("肿瘤科")
    if any(k in q for k in ["小儿", "宝宝", "儿童", "婴儿"]):
        inferred.add("儿科")
    if not inferred:
        # 若无匹配，则不加限制
        inferred = {"儿科", "肿瘤科"}
    return inferred


# ==============================================================================
#  数据加载：扫描目录并读入 CSV（新增返回 departments；实现保持不变）
# ==============================================================================
def load_medical_chunks(data_folder: str = 'Chinese-medical-dialogue-data-master') -> tuple[list[str], list[str], list[str]]:
    """遍历数据目录，读取各子目录的 CSV，组装成三元组并返回。

    Returns:
        chunks:    格式化后的『[Patient]/[Doctor]』问答文本列表
        sources:   每条问答对应的源 CSV 文件路径
        departments: 每条问答对应的科室名（优先取 CSV 内 department 列，否则取上级目录）

    说明：
        - 未更改你原本的编码优先顺序与异常处理，仅添加注释。
        - departments 的返回是为了方便后续写入 metadata 做按科室检索。
    """
    chunks, sources, departments = [], [], []
    encodings_to_try = ['gb18030', 'gbk', 'utf-8', 'gb2312']

    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            if not file.endswith('.csv'):
                continue

            filepath = os.path.join(subdir_path, file)
            print(f"📂 Reading file: {filepath}")

            success = False
            for enc in encodings_to_try:
                try:
                    with open(filepath, 'r', encoding=enc) as f:
                        reader = csv.DictReader(f)
                        count = 0
                        for row in reader:
                            q = row.get('ask', '').strip()
                            a = row.get('answer', '').strip()
                            title = (row.get('title') or '').strip()  # ✅ 新增

                            dept = (row.get('department') or '').strip() or os.path.basename(subdir_path)
                            if q and a:
                                chunk = f"[Title]: {title}\n[Patient]: {q}\n[Doctor]: {a}"  # ✅ 替换旧行
                                chunks.append(chunk)
                                sources.append(filepath)
                                departments.append(dept)
                                # 角色标签说明：帮助 LLM 更好区分提问者/回答者
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


# ==============================================================================
#  单文件数据载入（与主流程无强绑定，保持原实现）
# ==============================================================================
def load_dataset(file_path: str) -> list[str]:
    """从单个文本文件中按行读取内容（剔除空行），用于简单测试或演示。

    注意：此函数与 RAG 主流程相对独立，本项目主要用 CSV 批量载入。
    """
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = [line.strip() for line in file if line.strip()]
        print(f"✅ Loaded {len(dataset)} chunks from '{file_path}'")
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    return dataset


# ==============================================================================
#  生成文本嵌入（保持原逻辑，仅补充注释）
# ==============================================================================
def get_text_embedding(text: str, max_length: int = 800) -> list[float]:
    """调用 Ollama Embeddings 服务获取文本向量。

    - 超长文本会被截断至 max_length 以内（默认 800 字符），避免请求失败/效率低下。
    - 内置最多 3 次的指数退避重试。
    """
    if len(text) > max_length:
        text = text[:max_length]
        # 解释：避免嵌入 API 过长报错或低效

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
            sleep(2 ** attempt)  # 指数退避


# ==============================================================================
#  构建向量数据库（写入 metadata: source, department）
# ==============================================================================
def build_vector_db(chunks: list[str], sources: list[str], departments: list[str]):
    """对每条问答做嵌入，写入全局 VECTOR_DB。

    入参：
        chunks:        问答文本列表
        sources:       对应的源文件路径
        departments:   对应的科室名

    说明：
        - 保持你的实现；仅补充注释。
        - metadata 内写入 'source' 与 'department'，便于审计/按科室分析。
        - 注意：本文件保持对 normalize_dept(dept) 的调用；若该函数未定义会报错。
    """
    for i, (chunk, source, dept) in enumerate(zip(chunks, sources, departments), 1):
        try:
            embedding = get_text_embedding(chunk)
            VECTOR_DB.append({
                'id': i,
                'text': chunk,
                'embedding': np.array(embedding),
                'metadata': {
                    'source': source,
                    'department': normalize_dept(dept)  # 保持原有调用
                },
            })
            print(f"📌 Embedded chunk {i}/{len(chunks)} | Dept: {dept} | From: {source}")
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")


# ==============================================================================
#  相似度计算（余弦相似度）
# ==============================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两向量的余弦相似度（返回 [-1, 1]，1 表示最相似）。"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==============================================================================
#  检索：按相似度召回 + 软科室过滤/加权（逻辑保持不变）
# ==============================================================================
def retrieve(query: str, top_n: int = 5, score_threshold: float = 0.6) -> list[dict]:
    """
    功能：对用户输入 query 进行向量检索，返回最相关的若干回答片段。
    
    改进要点：
      1️⃣ 若初始召回为空，则自动降低阈值（例如从 0.6 → 0.5）再重试一次；
      2️⃣ 若按科室过滤后样本太少，改为使用全局向量池；
      3️⃣ 保留基础加权机制（命中科室的结果加少量 bonus 分）。
    """
    # ✅ 1. 利用 infer_departments 获取科室标签
    inferred = infer_departments(query)
    try:
        # ✅ 2. 根据关键词自动增强 query（软引导）
        DEPT_KEY_HINTS = {
            "肿瘤科": ["癌", "肿瘤", "放疗", "化疗", "术后", "恶性"],
            "儿科": ["儿童", "小孩", "宝宝", "中耳炎", "发烧"],
        }
        keywords = []
        for dept, hints in DEPT_KEY_HINTS.items():
            if any(h in query for h in hints):
                keywords.extend(hints[:2])  # 取前2个词增强
        if keywords:
            query = f"{query} {' '.join(keywords)}"

        # ✅ 3. 构造语义强化 query，引导模型识别问题领域
        query_with_hint = f"【可能科室：{', '.join(inferred)}】{query}"

        # ✅ 4. 生成 embedding 向量（使用语义强化后的 query）
        query_vector = np.array(get_text_embedding(query_with_hint))

    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        return []

    # === 1. 科室软过滤 ===
    if inferred:
        pool = [item for item in VECTOR_DB if item['metadata'].get('department') in inferred]
    else:
        pool = VECTOR_DB


    # 若过滤后样本太少（少于总量10%），放宽到全库，避免漏召回
    if len(pool) < 0.1 * len(VECTOR_DB):
        pool = VECTOR_DB

    # === 2. 初步计算相似度 ===
    results = []
    for item in pool:
        base = cosine_similarity(query_vector, item['embedding'])
        if base < score_threshold:
            continue

        dept = item['metadata'].get('department')
        dept = normalize_dept(dept) if dept else dept
        bonus = 0.05 if (inferred and dept in inferred) else 0.0
        score = base + bonus

        result = item.copy()
        result['score'] = score
        result['base_score'] = base
        results.append(result)

    # === 3. 若完全召不回结果，则自动降低阈值再试一次 ===
    if not results:
        print(f"⚠️ No results above {score_threshold:.2f}, retrying with threshold {score_threshold - 0.1:.2f}...")
        score_threshold -= 0.1
        for item in pool:
            base = cosine_similarity(query_vector, item['embedding'])
            if base < score_threshold:
                continue
            dept = item['metadata'].get('department')
            dept = normalize_dept(dept) if dept else dept
            bonus = 0.05 if (inferred and dept in inferred) else 0.0
            score = base + bonus
            result = item.copy()
            result['score'] = score
            result['base_score'] = base
            results.append(result)

    # === 4. 排序与返回 ===
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


# ==============================================================================
#  证据融合式回答：把 Top-K 片段列为 Evidence，并让 LLM 生成结构化结论
# ==============================================================================
def compose_answer(user_input: str, hits: list[dict]) -> str:
    """
    功能：整合检索结果，生成一个双语、结构化、可读性强的回答。

    输出结构：
    1️⃣ 📚 Retrieved context — 展示系统检索到的核心内容；
    2️⃣ 🧠 中文回答部分 — 模型生成的主要中文回答；
    3️⃣ 💡 English summary — 简短的英文总结。

    改进要点：
    ✅ 调整空行与缩进，使输出阅读体验更清晰；
    ✅ 限制 context 数量为 2，保证简洁；
    ✅ 输出一次性结构，避免重复段落。
    - 限制 Evidence 数量 ≤ 2，避免信息过载；
    - 中英文输出分区清晰；
    """
    # Step 1️⃣ 限制最多展示2条context
    if len(hits) > 2:
        print(f"ℹ️ Retrieved {len(hits)} results, truncating to top 2 for concise answer.")
        hits = hits[:2]

    # Step 2️⃣ 构造 context 块
    context_lines = []
    for i, item in enumerate(hits, 1):
        dept = item["metadata"].get("department", "未知科室")
        src = os.path.basename(item["metadata"].get("source", ""))
        score = f"{item.get('base_score', item.get('score', 0.0)):.2f}"
        text_preview = item["text"].strip().replace("\n", " ")
        context_lines.append(f"[{i}] ({dept}, score={score}, src={src})\n{text_preview}")
    context_block = "\n\n".join(context_lines)

    # Step 3️⃣ 构造 prompt
    system_prompt = f"""
你是一名谨慎的医学助理，请根据以下检索到的医疗问答内容，使用中英双语回答问题。
格式要求：
1️⃣ 输出两部分：
    🧠 中文回答部分：
    💡 English summary:
2️⃣ 用正式、简洁、专业的语气；
3️⃣ 不要编造或臆测；
4️⃣ 若信息不足，请指出不确定性。

---------------------
用户提问：
{user_input}

参考内容：
{context_block}
---------------------
""".strip()

    # Step 4️⃣ 调用模型
    try:
        resp = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            stream=False,
        )
        answer = resp["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ 生成失败：{e}"

    # Step 5️⃣ 清理空行并格式化输出
    # 统一空行结构，防止输出挤在一起或缩进错乱
    zh_part = ""
    en_part = ""

    if "💡 English summary" in answer:
        zh_part, en_part = answer.split("💡 English summary", 1)
        zh_part = zh_part.replace("🧠 中文回答部分：", "").strip()
        en_part = en_part.lstrip("\n").strip()
    else:
        zh_part = answer.strip()
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


# ==============================================================================
#  交互主循环（保留原有双输出：流式回答 + 证据融合回答）
# ==============================================================================
def run_chatbot():
    """
    功能：交互主循环（简化版）。
    改进要点：
      ✅ 仅保留结构化回答（compose_answer 输出），不再打印双份 Chatbot response；
      ✅ “📚 Retrieved context” 仅出现一次，由 compose_answer 统一生成；
      ✅ 确保输出顺序清晰、专业。
    """
    
    print("\n🤖 Welcome to the RAG Chatbot! Type your question or 'exit' to quit.\n")

    while True:
        user_input = input("Ask me a question: ")
        if user_input.strip().lower() == 'exit':
            print("👋 Goodbye!")
            break

        # Step 1️⃣: 检索最相关上下文
        top_chunks = retrieve(user_input, top_n=5)
        if not top_chunks:
            print("⚠️ No relevant information found.\n")
            continue

        # Step 2️⃣: 在 compose_answer 内统一生成完整输出（包含 Retrieved context）
        print("\n💬 Chatbot response:")
        fused = compose_answer(user_input, top_chunks)
        print(fused)


# ==============================================================================
#  主入口：加载数据 → 构建向量库 → 进入交互
# ==============================================================================
# === MAIN: Load, build, then chat ===
if __name__ == '__main__':
    print("📥 Loading data and building vector database...")

    chunks, sources, departments = load_medical_chunks(data_folder=DATA_DIR)

    if chunks:
        # ✅ 改进：稳定去重，保留加载顺序
        seen = set()
        unique_data = []
        for triplet in zip(chunks, sources, departments):
            if triplet not in seen:
                unique_data.append(triplet)
                seen.add(triplet)

        chunks, sources, departments = zip(*unique_data)

        build_vector_db(list(chunks), list(sources), list(departments))
        print("✅ Vector DB ready.")
        run_chatbot()
    else:
        print("❌ No data to build vector DB. Exiting.")

