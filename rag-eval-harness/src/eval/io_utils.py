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
            disease = str(obj.get("disease") or obj.get("dept") or obj.get("bucket") or obj.get("department") or "unknown")
            source_group = obj.get("source_group")
            q = obj.get("question") or obj.get("ask")
            if not q:
                raise ValueError(f"Missing question at line {i+1}")
            gold = obj.get("gold_answer") or obj.get("answer")
            qs.append(
                Question(
                    id=qid,
                    disease=disease,
                    question=str(q).strip(),
                    gold_answer=gold,
                    source_group=str(source_group).strip() if source_group else None,
                )
            )
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
    df.to_csv(p, index=False, encoding="utf-8-sig")

def snapshot_config(config: Dict[str, Any], run_args: Dict[str, Any], out_dir: str) -> None:
    selected_system_names = list(run_args.get("selected_system_names", run_args.get("systems", [])))
    systems_cfg = dict(config.get("systems", {}))
    effective_systems = {name: systems_cfg.get(name) for name in selected_system_names if name in systems_cfg}
    snap: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": sys.platform,
        "config": config,
        "run_args": run_args,
        "selected_system_names": selected_system_names,
        "effective_run": {
            "run": run_args.get("run"),
            "selected_system_names": selected_system_names,
            "seed": run_args.get("seed"),
            "limit": run_args.get("limit"),
            "top_n": run_args.get("top_n"),
            "score_threshold": run_args.get("score_threshold"),
            "rag_mode": run_args.get("rag_mode"),
            "rag_base_url": run_args.get("rag_base_url"),
            "rag_auto_index": run_args.get("rag_auto_index"),
        },
        "effective_systems": effective_systems,
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
        groups[(r.system_name, r.disease)].append(r)

    def is_refusal(text: str) -> bool:
        t = text.lower()
        phrases = ["insufficient info", "consult a doctor", "see a doctor", "visit a doctor", "无法", "建议就医", "请咨询医生"]
        return any(p in t for p in phrases)

    rows: List[Dict[str, Any]] = []
    for (system, disease), recs in groups.items():
        top_scores=[]; mean_scores=[]; rc=[]; refusals=0; answer_scores=[]
        for r in recs:
            rc.append(int(r.meta.get("retrieval_count", 0)))
            scores=[float(c.get("score",0.0)) for c in (r.contexts or [])]
            top_scores.append(max(scores) if scores else 0.0)
            mean_scores.append(sum(scores)/len(scores) if scores else 0.0)
            if is_refusal(r.answer):
                refusals += 1
            if r.answer_similarity is not None:
                answer_scores.append(float(r.answer_similarity))
        n=len(recs)
        rows.append({
            "system": system,
            "disease": disease,
            "count": n,
            "retrieval_count_avg": float(np.mean(rc)) if n else 0.0,
            "top_score_avg": float(np.mean(top_scores)) if n else 0.0,
            "avg_score_avg": float(np.mean(mean_scores)) if n else 0.0,
            "refusal_rate": float(refusals/n) if n else 0.0,
            "answer_similarity_avg": float(np.mean(answer_scores)) if answer_scores else None,
            "answer_scored_rate": float(len(answer_scores)/n) if n else 0.0,
            "answer_similarity_std": float(np.std(answer_scores)) if answer_scores else None,
            "answer_similarity_min": float(np.min(answer_scores)) if answer_scores else None,
            "answer_similarity_max": float(np.max(answer_scores)) if answer_scores else None,
        })
    return rows
