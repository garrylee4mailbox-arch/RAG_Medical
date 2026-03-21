#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Selection:
    row_index: int


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "Chinese-medical-dialogue-data-master"
OUTPUT_PATH = REPO_ROOT / "data" / "questions" / "smoke_v2.0.jsonl"
SOURCE_GROUPS: Tuple[str, ...] = ("Oncology", "Pediatric")
SMOKE_SAMPLE_SIZE = 10


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\r", " ").replace("\n", " ")).strip()


def normalize_department(raw_department: str) -> str:
    value = normalize_text(raw_department).lower()
    if value in {"oncology"} or "oncology" in value:
        return "oncology"
    if value in {"pediatric", "pediatrics"} or "pediatric" in value:
        return "pediatrics"
    return "unknown"


def find_source_csv(department: str) -> Path:
    matches = [path for path in SOURCE_ROOT.iterdir() if path.is_dir() and department in path.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source folder for {department}, found {len(matches)}")
    csv_files = sorted(matches[0].glob("*.csv"))
    if len(csv_files) != 1:
        raise FileNotFoundError(f"Expected one CSV in {matches[0]}, found {len(csv_files)}")
    return csv_files[0]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="gb18030", newline="") as handle:
        return list(csv.DictReader(handle))


def make_record(
    source_csv: Path,
    row: Dict[str, str],
    selection: Selection,
    source_group: str,
    ordinal: int,
) -> Dict[str, object]:
    question = normalize_text(row.get("ask", ""))
    gold_answer = normalize_text(row.get("answer", ""))
    disease = normalize_text(row.get("department", ""))
    title = normalize_text(row.get("title", ""))
    if not question:
        raise ValueError(f"Selected row {selection.row_index} in {source_csv} has empty 'ask'")
    if not gold_answer:
        raise ValueError(f"Selected row {selection.row_index} in {source_csv} has empty 'answer'")
    if not disease:
        raise ValueError(f"Selected row {selection.row_index} in {source_csv} has empty 'department'")

    prefix = "oncology" if source_group == "Oncology" else "pediatric"
    return {
        "id": f"smoke_v2_{prefix}_{ordinal:03d}",
        "question": question,
        "gold_answer": gold_answer,
        "disease": disease,
        "source_group": normalize_department(source_group),
        "source_csv": source_csv.relative_to(PROJECT_ROOT).as_posix(),
        "source_row_id": selection.row_index,
        "title": title,
    }


def build_records() -> List[Dict[str, object]]:
    # Smoke is limited to 10 randomly sampled questions total for faster runs.
    records: List[Dict[str, object]] = []
    candidates: List[Tuple[str, Path, Selection, Dict[str, str]]] = []

    for source_group in SOURCE_GROUPS:
        source_csv = find_source_csv(source_group)
        rows = load_rows(source_csv)
        for row_index, row in enumerate(rows):
            candidates.append((source_group, source_csv, Selection(row_index), row))

    if len(candidates) < SMOKE_SAMPLE_SIZE:
        raise ValueError(f"Expected at least {SMOKE_SAMPLE_SIZE} candidate rows, found {len(candidates)}")

    ordinals: Dict[str, int] = {}
    for source_group, source_csv, selection, row in random.sample(candidates, SMOKE_SAMPLE_SIZE):
        ordinals[source_group] = ordinals.get(source_group, 0) + 1
        records.append(
            make_record(
                source_csv=source_csv,
                row=row,
                selection=selection,
                source_group=source_group,
                ordinal=ordinals[source_group],
            )
        )

    return records


def write_jsonl(records: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: List[Dict[str, object]]) -> None:
    print("Source CSV files:")
    for source_group in SOURCE_GROUPS:
        matching_record = next(
            (record for record in records if record["source_group"] == normalize_department(source_group)),
            None,
        )
        if matching_record is not None:
            print(f"  - {matching_record['source_csv']}")

    print("Selection counts:")
    for source_group in SOURCE_GROUPS:
        count = sum(1 for record in records if record["source_group"] == normalize_department(source_group))
        print(f"  - {source_group}: {count}")

    print("Diseases:")
    for source_group in SOURCE_GROUPS:
        values = sorted({str(record["disease"]) for record in records if record["source_group"] == normalize_department(source_group)})
        print(f"  - {source_group}: {', '.join(values)}")

    print(f"Output: {OUTPUT_PATH.resolve()}")


def main() -> None:
    records = build_records()
    if len(records) != SMOKE_SAMPLE_SIZE:
        raise ValueError(f"Expected {SMOKE_SAMPLE_SIZE} total records, found {len(records)}")
    write_jsonl(records, OUTPUT_PATH)
    summarize(records)


if __name__ == "__main__":
    main()
