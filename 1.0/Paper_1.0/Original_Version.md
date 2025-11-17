# 🧪 Medical RAG Chatbot - 学习笔记版
本 Notebook 是对 `LATEST.py` 的重构与分解。  
每个步骤分成 **代码单元 + 解释单元**，便于学习与记录。


## 1. 导入依赖库
这里我们用到的库：
- `numpy`：处理向量与相似度计算
- `pandas`：读取数据时备用
- `csv`：处理医疗对话数据文件
- `ollama`：调用大模型和 embedding
- `time.sleep`：做失败重试时的等待



```python
import numpy as np
import pandas as pd
import ollama
from time import sleep
import os
import csv
```

## 2. 全局配置
- **EMBEDDING_MODEL**：用来生成向量的模型
- **LANGUAGE_MODEL**：对话时使用的语言模型
- **VECTOR_DB**：存放所有文本块的向量数据库



```python
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3'
VECTOR_DB = []  # 存 embeddings + 原始文本 + 来源信息

```

## 3. 加载医疗数据
函数 `load_medical_chunks()`：
- 遍历数据文件夹（默认：`Chinese-medical-dialogue-data-master`）
- 读取 CSV 文件中的问答
- 格式化为 `[Patient]` 和 `[Doctor]`
- 返回文本块 + 来源路径




```python
def load_medical_chunks(data_folder='Chinese-medical-dialogue-data-master') -> tuple[list[str], list[str]]:
    """
    Traverse through all subdirectories, read CSV files, and return:
    - chunks: list of "Q: ...\nA: ..." texts with labels
    - sources: list of file paths for each chunk
    """
    chunks = []
    sources = []
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
                            if q and a:
                                
                                chunk = f"[Patient]: {q}\n[Doctor]: {a}"
                                chunks.append(chunk)
                                sources.append(filepath)
                                '''
                                Explanation:
                                Each QA pair is now labeled explicitly as [Patient] and [Doctor], 
                                which helps the LLM better understand the role-based structure of 
                                the conversation
                                Benefit:
                                Improves context understanding by the language model.
                                Helps disambiguate who is asking vs answering during generation.
                                '''
                                count += 1
                        print(f"✅ Loaded {count} Q&A pairs with encoding: {enc}")
                        success = True
                        break
                except Exception as e:
                    print(f"❌ Failed with encoding {enc}: {e}")

            if not success:
                print(f"❌ Could not read file: {filepath} with any known encoding.")

    print(f"\n✅ Total loaded QA chunks: {len(chunks)}")
    return chunks, sources
```

## 4. 文本嵌入 (Embedding)
函数 `get_text_embedding()`：
- 调用 Ollama API 获取文本向量
- 长文本自动截断到 800 字符
- 失败会重试，最多 3 次



```python
def get_text_embedding(text: str, max_length: int = 800) -> list[float]:
    if len(text) > max_length:
        text = text[:max_length]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
            sleep(2 ** attempt)  # 退避重试

```

## 5. 构建向量数据库
函数 `build_vector_db()`：
- 对每个文本块生成 embedding
- 存入 `VECTOR_DB`
- 结构：{id, text, embedding, metadata}



```python
def build_vector_db(chunks: list[str], sources: list[str]):
    for i, (chunk, source) in enumerate(zip(chunks, sources), 1):
        try:
            embedding = get_text_embedding(chunk)
            VECTOR_DB.append({
                'id': i,
                'text': chunk,
                'embedding': np.array(embedding),
                'metadata': {'source': source}
            })
            print(f"📌 Embedded chunk {i}/{len(chunks)} | From: {source}")
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")

```

## 6. 余弦相似度
计算两个向量的相似度，范围在 -1 到 1：
- 1 = 完全相似
- 0 = 无关
- -1 = 相反



```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

```

## 7. 信息检索
函数 `retrieve()`：
- 把用户 query 转换为 embedding
- 与数据库里所有文本块计算相似度
- 过滤低于阈值的结果（默认 0.6）
- 返回最相关的前 N 个



```python
def retrieve(query: str, top_n: int = 5, score_threshold: float = 0.6) -> list[dict]:
    try:
        query_vector = np.array(get_text_embedding(query))
    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        return []

    results = []
    for item in VECTOR_DB:
        score = cosine_similarity(query_vector, item['embedding'])
        if score >= score_threshold:
            result = item.copy()
            result['score'] = score
            results.append(result)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]

```

## 8. 聊天机器人主循环
函数 `run_chatbot()`：
- 接收用户输入
- 调用 `retrieve()` 找相关文本
- 构造 Prompt → 传给语言模型
- 输出回答



```python
def run_chatbot():
    print("\n🤖 Welcome to the RAG Chatbot! Type your question or 'exit' to quit.\n")

    while True:
        user_input = input("Ask me a question: ")
        if user_input.strip().lower() == 'exit':
            print("👋 Goodbye!")
            break

        top_chunks = retrieve(user_input, top_n=3)

        if not top_chunks:
            print("⚠️ No relevant information found.\n")
            continue

        print("\n📚 Retrieved context:")
        for item in top_chunks:
            print(f" - (similarity: {item['score']:.2f}) {item['text']}")

        context_str = '\n'.join([f" - {item['text']}" for item in top_chunks])
        system_prompt = f"""
        You are a helpful and cautious medical assistant.
        Use ONLY the following context to answer the user’s question.
        DO NOT make up or infer facts not in the context.
        Context:
        {context_str}
        """

        print("\n💬 Chatbot response:")
        try:
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_input},
                ],
                stream=True,
            )
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print()
        except Exception as e:
            print(f"❌ Error during chat: {e}")

```

## 9. 主程序入口
- 加载数据
- 构建向量数据库
- 启动聊天机器人




```python
if __name__ == '__main__':
    print("📥 Loading data and building vector database...")

    chunks, sources = load_medical_chunks(data_folder='Chinese-medical-dialogue-data-master')

    if chunks:
        unique_data = list(set(zip(chunks, sources)))
        chunks, sources = zip(*unique_data)
        build_vector_db(list(chunks), list(sources))
        print("✅ Vector DB ready.")
        run_chatbot()
    else:
        print("❌ No data to build vector DB. Exiting.")

```

    📥 Loading data and building vector database...
    


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[10], line 4
          1 if __name__ == '__main__':
          2     print("📥 Loading data and building vector database...")
    ----> 4     chunks, sources = load_medical_chunks(data_folder='Chinese-medical-dialogue-data-master')
          6     if chunks:
          7         unique_data = list(set(zip(chunks, sources)))
    

    Cell In[4], line 7, in load_medical_chunks(data_folder)
          4 sources = []
          5 encodings_to_try = ['gb18030', 'gbk', 'utf-8', 'gb2312']
    ----> 7 for subdir in os.listdir(data_folder):
          8     subdir_path = os.path.join(data_folder, subdir)
          9     if not os.path.isdir(subdir_path):
    

    FileNotFoundError: [WinError 3] 系统找不到指定的路径。: 'Chinese-medical-dialogue-data-master'



```python

```
