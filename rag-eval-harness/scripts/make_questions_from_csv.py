#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Optional
import pandas as pd

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make JSONL questions from CSVs")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-n", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--field-question", default="ask")
    p.add_argument("--field-answer")
    p.add_argument("--bucket")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    dfs = [pd.read_csv(x) for x in args.input]
    df = pd.concat(dfs, ignore_index=True)
    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.seed)

    qcol = args.field_question if args.field_question in df.columns else ("question" if "question" in df.columns else None)
    if qcol is None:
        raise ValueError("Cannot find question column. Use --field-question.")
    acol = args.field_answer if args.field_answer and args.field_answer in df.columns else None

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            rec = {"id": f"q{i+1:04d}", "bucket": args.bucket or str(row.get("bucket","default")), "question": str(row[qcol])}
            if acol:
                rec["gold_answer"] = str(row[acol])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df)} questions -> {out}")

if __name__ == "__main__":
    main()
