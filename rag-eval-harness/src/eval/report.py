from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import List
from .schemas import AnswerRecord
from .io_utils import aggregate_metrics

def generate_report(run_dir: str, records: List[AnswerRecord]) -> None:
    out = Path(run_dir)/"report.md"
    metrics = aggregate_metrics(records)
    by_system = defaultdict(list)
    for r in records:
        by_system[r.system_name].append(r)

    lines=[]
    lines.append(f"# Evaluation Report: {Path(run_dir).name}\n")
    lines.append("## Disease-level Metrics\n")
    lines.append("These metrics diagnose retrieval behavior: how many contexts were returned, how strong the retrieved scores were, and how often the system refused.\n")
    if metrics:
        headers=["System","Disease","Count","AvgContexts","AvgTopScore","AvgMeanScore","RefusalRate"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")
        for m in metrics:
            lines.append(f"| {m['system']} | {m['disease']} | {m['count']} | {m['retrieval_count_avg']:.2f} | {m['top_score_avg']:.2f} | {m['avg_score_avg']:.2f} | {m['refusal_rate']:.2f} |")
        lines.append("")
    else:
        lines.append("No metrics.\n")

    lines.append("## Per-disease Evaluation\n")
    lines.append("These metrics evaluate final generated answer quality by embedding similarity between the generated answer and the gold answer.\n")
    if metrics:
        headers=["System","Disease","Count","ScoredRate","AnswerSimAvg","AnswerSimStd","AnswerSimMin","AnswerSimMax"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")
        for m in metrics:
            avg = "-" if m["answer_similarity_avg"] is None else f"{m['answer_similarity_avg']:.3f}"
            std = "-" if m["answer_similarity_std"] is None else f"{m['answer_similarity_std']:.3f}"
            min_v = "-" if m["answer_similarity_min"] is None else f"{m['answer_similarity_min']:.3f}"
            max_v = "-" if m["answer_similarity_max"] is None else f"{m['answer_similarity_max']:.3f}"
            lines.append(f"| {m['system']} | {m['disease']} | {m['count']} | {m['answer_scored_rate']:.2f} | {avg} | {std} | {min_v} | {max_v} |")
        lines.append("")
    else:
        lines.append("No metrics.\n")

    lines.append("## Samples (Rule-based)\n")
    for sysname, recs in by_system.items():
        lines.append(f"### {sysname}\n")
        # show a few samples
        for r in recs[:3]:
            sim = "-" if r.answer_similarity is None else f"{r.answer_similarity:.3f}"
            lines.append(f"- **Q:** {r.question}\n  - **A:** {r.answer[:220].replace('\\n',' ')}\n  - **Answer Similarity:** {sim}\n")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
