from __future__ import annotations

import argparse, os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from ..rag import rag_adapter
from ..rag.rag_api_client import RagApiClient
from ..rag import rag_config as rag_defaults
from ..llm.ollama_client import OllamaClient
from ..llm.openai_client import OpenAIClient
from .schemas import AnswerRecord
from .io_utils import load_questions, write_jsonl, write_csv, snapshot_config, aggregate_metrics
from .report import generate_report

def parse_bool(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    t = s.strip().lower()
    if t in ("1","true","yes","y","t"): return True
    if t in ("0","false","no","n","f"): return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {s}")

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Medical RAG evaluation harness (deterministic).")
    ap.add_argument("--run", required=True, help="smoke|full")
    ap.add_argument("--systems", help="Override systems list, comma-separated")
    ap.add_argument("--out", help="Output dir (default runs/<auto>)")
    ap.add_argument("--seed", type=int, help="Seed for deterministic sampling")
    ap.add_argument("--limit", type=int, help="Limit questions")
    ap.add_argument("--top-n", dest="top_n", type=int, help="Override RAG top_n")
    ap.add_argument("--score-threshold", dest="score_threshold", type=float, help="Override RAG score_threshold")
    ap.add_argument("--rag-mode", choices=["inproc","http"], help="Override rag mode")
    ap.add_argument("--rag-base-url", help="Override rag base url")
    ap.add_argument("--rag-auto-index", help="Override rag auto index (true/false)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    systems_cfg = yaml.safe_load((repo_root/"configs/systems.yaml").read_text(encoding="utf-8"))
    runs_cfg = yaml.safe_load((repo_root/"configs/runs.yaml").read_text(encoding="utf-8"))

    if args.run not in runs_cfg:
        raise ValueError(f"Run '{args.run}' not in configs/runs.yaml")
    run_cfg = runs_cfg[args.run]

    system_names = [s.strip() for s in args.systems.split(",")] if args.systems else list(run_cfg.get("systems", []))
    if "openai_llm_only" in system_names and not os.getenv("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY not set; skip openai_llm_only")
        system_names = [s for s in system_names if s != "openai_llm_only"]

    rag_mode = args.rag_mode or run_cfg.get("rag_mode") or systems_cfg["rag"].get("mode", "http")
    rag_base_url = args.rag_base_url or systems_cfg["rag"].get("base_url", "http://127.0.0.1:8008")
    rag_auto_index = parse_bool(args.rag_auto_index) if args.rag_auto_index is not None else run_cfg.get("rag_auto_index", systems_cfg["rag"].get("auto_index", True))
    top_n = args.top_n or systems_cfg["rag"].get("top_n", rag_defaults.DEFAULT_TOP_N)
    score_threshold = args.score_threshold if args.score_threshold is not None else systems_cfg["rag"].get("score_threshold", rag_defaults.DEFAULT_SCORE_THRESHOLD)

    out_dir = Path(args.out) if args.out else (repo_root/"runs"/f"{args.run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    questions_path = repo_root / run_cfg["questions"]
    questions = load_questions(str(questions_path), limit=args.limit, seed=args.seed)
    if not questions:
        print("No questions loaded.")
        return

    rag_client: Optional[RagApiClient] = None
    if "rag" in system_names:
        data_sources = systems_cfg["rag"].get("data_sources", rag_defaults.DEFAULT_DATA_SOURCES)
        cache_dir = systems_cfg["rag"].get("cache_dir", None)
        if rag_mode == "inproc":
            print(f"[INFO] prepare_rag inproc: {data_sources}")
            rag_adapter.prepare_rag(data_sources=data_sources, cache_dir=cache_dir)
        else:
            rag_client = RagApiClient(rag_base_url)
            if rag_auto_index:
                print(f"[INFO] ensure_index via http: {data_sources}")
                rag_client.ensure_index(data_sources=data_sources, cache_dir=cache_dir, force_rebuild=False)

    ollama = None
    if "ollama_llm_only" in system_names:
        c = systems_cfg["ollama_llm_only"]
        ollama = OllamaClient(model=c.get("model","llama3"), temperature=c.get("temperature",0.0), max_tokens=c.get("max_tokens",256), base_url=c.get("base_url"))

    openai_cli = None
    if "openai_llm_only" in system_names:
        c = systems_cfg["openai_llm_only"]
        openai_cli = OpenAIClient(model=c.get("model","gpt-3.5-turbo"), temperature=c.get("temperature",0.0), max_tokens=c.get("max_tokens",256), base_url=c.get("base_url"))

    records: List[AnswerRecord] = []
    pbar = tqdm(total=len(questions)*len(system_names), desc="Evaluating", unit="answer")
    for q in questions:
        for sysname in system_names:
            if sysname == "rag":
                if rag_mode == "inproc":
                    res = rag_adapter.rag_answer(q.question, top_n=top_n, score_threshold=score_threshold)
                    # ensure latency keys exist (split not available -> best effort)
                    res["meta"].setdefault("latency_ms_total", 0.0)
                    res["meta"].setdefault("latency_ms_retrieve", 0.0)
                    res["meta"].setdefault("latency_ms_generate", res["meta"].get("latency_ms_total", 0.0))
                else:
                    assert rag_client is not None
                    res = rag_client.answer(q.question, top_n=top_n, score_threshold=score_threshold)
                    res["meta"].setdefault("latency_ms_total", 0.0)
                    res["meta"].setdefault("latency_ms_retrieve", 0.0)
                    res["meta"].setdefault("latency_ms_generate", res["meta"].get("latency_ms_total", 0.0))
                records.append(AnswerRecord(run_id=run_id, question_id=q.id, bucket=q.bucket, question=q.question, system_name="rag",
                                            answer=res.get("answer",""), contexts=res.get("contexts",[]), meta=res.get("meta",{})))
            elif sysname == "ollama_llm_only":
                assert ollama is not None
                ans, meta = ollama.generate(q.question)
                meta.setdefault("retrieval_count", 0)
                records.append(AnswerRecord(run_id=run_id, question_id=q.id, bucket=q.bucket, question=q.question, system_name=sysname,
                                            answer=ans, contexts=[], meta=meta))
            elif sysname == "openai_llm_only":
                if openai_cli is None: 
                    pbar.update(1); 
                    continue
                ans, meta = openai_cli.generate(q.question)
                meta.setdefault("retrieval_count", 0)
                records.append(AnswerRecord(run_id=run_id, question_id=q.id, bucket=q.bucket, question=q.question, system_name=sysname,
                                            answer=ans, contexts=[], meta=meta))
            pbar.update(1)
    pbar.close()

    write_jsonl([r.__dict__ for r in records], str(out_dir/"answers.jsonl"))
    write_csv(aggregate_metrics(records), str(out_dir/"metrics.csv"))
    snapshot_config({"systems": systems_cfg, "runs": runs_cfg}, {
        "run": args.run,
        "systems": system_names,
        "seed": args.seed,
        "limit": args.limit,
        "top_n": top_n,
        "score_threshold": score_threshold,
        "rag_mode": rag_mode,
        "rag_base_url": rag_base_url,
        "rag_auto_index": rag_auto_index,
    }, str(out_dir))
    generate_report(str(out_dir), records)
    print(f"[DONE] Results saved to {out_dir}")

if __name__ == "__main__":
    main()
