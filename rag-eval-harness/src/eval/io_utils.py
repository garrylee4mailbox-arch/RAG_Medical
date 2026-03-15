from __future__ import annotations

import json, sys, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
from .schemas import Question, AnswerRecord

def load_questions(path: str, limit: Optional[int] = None, seed: Optional[int] = None) -> List[Question]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    qs: List[Question] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id") or f"q{i+1:04d}")
            bucket = str(obj.get("bucket","default"))
            q = obj.get("question") or obj.get("ask")
            if not q:
                raise ValueError(f"Missing question at line {i+1}")
            gold = obj.get("gold_answer") or obj.get("answer")
            qs.append(Question(id=qid, bucket=bucket, question=str(q).strip(), gold_answer=gold))
    if limit is not None and limit < len(qs):
        import random
        rnd = random.Random(seed)
        rnd.shuffle(qs)
        qs = qs[:limit]
    return qs

def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(rows: Iterable[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(list(rows))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def snapshot_config(config: Dict[str, Any], run_args: Dict[str, Any], out_dir: str) -> None:
    snap: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": sys.platform,
        "config": config,
        "run_args": run_args,
    }
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=Path(out_dir).resolve().parents[1]).decode().strip()
        snap["git_commit"] = commit
    except Exception:
        pass
    p = Path(out_dir)/"config.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)

def aggregate_metrics(records: List[AnswerRecord]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    import numpy as np
    groups = defaultdict(list)
    for r in records:
        groups[(r.system_name, r.bucket)].append(r)

    def is_refusal(text: str) -> bool:
        t = text.lower()
        phrases = ["insufficient info", "consult a doctor", "see a doctor", "visit a doctor", "无法", "建议就医", "请咨询医生"]
        return any(p in t for p in phrases)

    rows: List[Dict[str, Any]] = []
    for (system, bucket), recs in groups.items():
        top_scores=[]; mean_scores=[]; rc=[]; refusals=0
        for r in recs:
            rc.append(int(r.meta.get("retrieval_count", 0)))
            scores=[float(c.get("score",0.0)) for c in (r.contexts or [])]
            top_scores.append(max(scores) if scores else 0.0)
            mean_scores.append(sum(scores)/len(scores) if scores else 0.0)
            if is_refusal(r.answer):
                refusals += 1
        n=len(recs)
        rows.append({
            "system": system,
            "bucket": bucket,
            "count": n,
            "retrieval_count_avg": float(np.mean(rc)) if n else 0.0,
            "top_score_avg": float(np.mean(top_scores)) if n else 0.0,
            "avg_score_avg": float(np.mean(mean_scores)) if n else 0.0,
            "refusal_rate": float(refusals/n) if n else 0.0,
        })
    return rows
