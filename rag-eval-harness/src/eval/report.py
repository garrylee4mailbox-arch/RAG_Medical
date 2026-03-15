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
    lines.append("## Summary Metrics\n")
    if metrics:
        headers=["System","Bucket","Count","AvgContexts","AvgTopScore","AvgMeanScore","RefusalRate"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")
        for m in metrics:
            lines.append(f"| {m['system']} | {m['bucket']} | {m['count']} | {m['retrieval_count_avg']:.2f} | {m['top_score_avg']:.2f} | {m['avg_score_avg']:.2f} | {m['refusal_rate']:.2f} |")
        lines.append("")
    else:
        lines.append("No metrics.\n")

    lines.append("## Samples (Rule-based)\n")
    for sysname, recs in by_system.items():
        lines.append(f"### {sysname}\n")
        # show a few samples
        for r in recs[:3]:
            lines.append(f"- **Q:** {r.question}\n  - **A:** {r.answer[:220].replace('\\n',' ')}\n")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
