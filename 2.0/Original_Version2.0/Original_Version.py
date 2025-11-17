import numpy as np
import ollama
from time import sleep

# === CONFIGURATION ===
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3'
TEXT_FILE = 'cat-facts.txt'
VECTOR_DB = []  # Will hold all embeddings and text chunks


import os
import pandas as pd
#===Second Phrase: Step 1: Embedding the medical data into chunks =====
import csv
import os

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



# === STEP 1: Load the dataset ===
def load_dataset(file_path: str) -> list[str]:
    """
    Load and clean text chunks from a file.
    Removes empty lines and strips whitespace.
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



# === STEP 2: Generate embeddings ===
def get_text_embedding(text: str, max_length: int = 800) -> list[float]:
    """
    Get embedding vector from Ollama for a given text chunk.
    Retries on failure.
    """
    if len(text) > max_length:
        text = text[:max_length]
        '''
        Explanation:
        To avoid embedding errors or inefficiencies,
        long text chunks are truncated to 800 characters (default).
        Benefit:
        Prevents request errors in the Ollama embedding API.
        Improves response speed and reliability during vector computation.
        '''
        
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
            sleep(2 ** attempt)  # Exponential backoff


# === STEP 3: Build vector database ===
def build_vector_db(chunks: list[str], sources: list[str]):
    """
    For each text chunk, compute embedding and store it in VECTOR_DB.
    Each entry includes: id, original text, its embedding vector, and file metadata.
    
    参数说明：
    - chunks: 问答文本列表
    - sources: 与 chunks 同顺序的来源文件路径列表
    """
    for i, (chunk, source) in enumerate(zip(chunks, sources), 1):
        try:
            embedding = get_text_embedding(chunk)
            VECTOR_DB.append({
                'id': i,
                'text': chunk,
                'embedding': np.array(embedding),
                'metadata': {'source': source},             
            })
            
                # Explanation:
                # Every chunk now stores its original source path (metadata['source']). 
                # This supports future traceability, debugging, and dataset analysis.
                # Benefit:
                # Enables dataset auditing (e.g., find which file caused irrelevant results).
                # Improves transparency of retrieved responses.
                
            print(f"📌 Embedded chunk {i}/{len(chunks)} | From: {source}")
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")


# === STEP 4: Cosine similarity function ===
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a score between -1 and 1 (1 = most similar).
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# === STEP 5: Retrieve top-N relevant chunks ===
def retrieve(query: str, top_n: int = 5, score_threshold: float = 0.6) -> list[dict]:
    """
    Given a query, embed it and return the top-N most similar chunks from the database.
    Filters out chunks below the similarity threshold.
    
    参数说明：
    - query: 用户输入的问题
    - top_n: 返回前几个最相似的文本块
    - score_threshold: 相似度最低要求（避免不相关内容）
    """
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
        '''
        Explanation:
        A minimum cosine similarity threshold (default: 0.6) is applied. 
        Irrelevant chunks below this threshold are filtered out.
        Benefit:
        Prevents hallucinated or unrelated responses.
        Keeps the chatbot focused only on highly relevant context.
        '''
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]

# === STEP 6: Chat loop ===
def run_chatbot():
    """
    Main interactive loop. Accepts user input, retrieves context, and sends to LLM.
    """
    print("\n🤖 Welcome to the RAG Chatbot! Type your question or 'exit' to quit.\n")

    while True:
        user_input = input("Ask me a question: ")
        if user_input.strip().lower() == 'exit':
            print("👋 Goodbye!")
            break

        # Retrieve most relevant context
        top_chunks = retrieve(user_input, top_n=3)

        if not top_chunks:
            print("⚠️ No relevant information found.\n")
            continue

        # Show retrieved knowledge to user
        print("\n📚 Retrieved context:")
        for item in top_chunks:
            print(f" - (similarity: {item['score']:.2f}) {item['text']}")

        # Construct prompt for LLM
        context_str = '\n'.join([f" - {item['text']}" for item in top_chunks])
        system_prompt = f"""
        You are a helpful and cautious medical assistant.
        Use ONLY the following context to answer the user’s question.
        DO NOT make up or infer facts that are not present in the context.
        If you don’t find enough information, reply: "I cannot find relevant information based on the current data."
        Context:
        {context_str}
        """
        # Explanation:
        # The system prompt was rewritten to strictly instruct the model to avoid hallucination and only use retrieved content.
        # Benefit:
        # Greatly reduces medically unsafe responses.
        # Ensures academic and professional quality in answers.
        # Generate answer using Ollama chat
        
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
            print()  # newline after response
        except Exception as e:
            print(f"❌ Error during chat: {e}")

# === MAIN: Load, build, then chat ===
if __name__ == '__main__':
    print("📥 Loading data and building vector database...")

    chunks, sources = load_medical_chunks(data_folder='Chinese-medical-dialogue-data-master')

    if chunks:
        # 去重处理（可选）
        unique_data = list(set(zip(chunks, sources)))
        chunks, sources = zip(*unique_data)
        '''
        Explanation:
        Before embedding, duplicate QA entries are removed. 
        This reduces redundancy in the vector database.
        Benefit:
        Reduces noise in retrieval.
        Speeds up embedding and retrieval steps.
        Improves diversity of retrieved answers.
        '''
        build_vector_db(list(chunks), list(sources))
        print("✅ Vector DB ready.")
        run_chatbot()
    else:
        print("❌ No data to build vector DB. Exiting.")
