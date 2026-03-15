# Question Sets

JSONL 每行一个对象，至少包含：
- id (可选，缺省自动生成)
- bucket
- question
- gold_answer (可选，Phase 1 不用于打分，仅保存)

用 `scripts/make_questions_from_csv.py` 从 CSV 生成新的 questions 文件。
