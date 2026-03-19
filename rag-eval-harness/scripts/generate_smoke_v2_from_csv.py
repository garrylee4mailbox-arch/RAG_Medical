#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Selection:
    row_index: int
    bucket: str


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "Chinese-medical-dialogue-data-master"
OUTPUT_PATH = REPO_ROOT / "data" / "questions" / "smoke_v2.0.jsonl"

# Fixed row indices keep regeneration deterministic and directly traceable to
# the original CSVs. The output uses the raw ask/answer fields with whitespace
# normalization only.
SELECTIONS: Dict[str, List[Selection]] = {
    "Oncology": [
        Selection(0, "symptoms"),
        Selection(1, "treatment"),
        Selection(22, "supportive_care"),
        Selection(35, "diagnosis"),
        Selection(44, "chemotherapy"),
        Selection(62, "diet"),
        Selection(80, "diet"),
        Selection(88, "symptoms"),
        Selection(97, "treatment"),
        Selection(184, "prognosis"),
    ],
    "Pediatric": [
        Selection(0, "weight_management"),
        Selection(74, "ent_treatment"),
        Selection(188, "digestive_care"),
        Selection(194, "neonatal_treatment"),
        Selection(195, "neonatal_symptoms"),
        Selection(197, "digestive_care"),
        Selection(198, "nutrition_deficiency"),
        Selection(199, "neuro_treatment"),
        Selection(201, "neuro_diagnosis"),
        Selection(202, "ent_symptoms"),
    ],
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\r", " ").replace("\n", " ")).strip()


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
    department: str,
    ordinal: int,
) -> Dict[str, object]:
    question = normalize_text(row.get("ask", ""))
    gold_answer = normalize_text(row.get("answer", ""))
    title = normalize_text(row.get("title", ""))
    if not question or not gold_answer:
        raise ValueError(f"Selected row {selection.row_index} in {source_csv} has empty ask/answer")

    prefix = "oncology" if department == "Oncology" else "pediatric"
    return {
        "id": f"smoke_v2_{prefix}_{ordinal:03d}",
        "question": question,
        "gold_answer": gold_answer,
        "bucket": selection.bucket,
        "source_csv": source_csv.relative_to(PROJECT_ROOT).as_posix(),
        "source_row_id": selection.row_index,
        "department": department,
        "title": title,
    }


def build_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen_sources = set()

    for department, selections in SELECTIONS.items():
        source_csv = find_source_csv(department)
        rows = load_rows(source_csv)

        for ordinal, selection in enumerate(selections, start=1):
            if selection.row_index >= len(rows):
                raise IndexError(f"Row {selection.row_index} is out of range for {source_csv}")
            source_key = (department, selection.row_index)
            if source_key in seen_sources:
                raise ValueError(f"Duplicate source row selected: {department} row {selection.row_index}")
            seen_sources.add(source_key)
            records.append(
                make_record(
                    source_csv=source_csv,
                    row=rows[selection.row_index],
                    selection=selection,
                    department=department,
                    ordinal=ordinal,
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
    for department in SELECTIONS:
        first_record = next(record for record in records if record["department"] == department)
        print(f"  - {first_record['source_csv']}")

    print("Selection counts:")
    for department in SELECTIONS:
        count = sum(1 for record in records if record["department"] == department)
        print(f"  - {department}: {count}")

    print("Buckets:")
    for department in SELECTIONS:
        values = sorted({str(record["bucket"]) for record in records if record["department"] == department})
        print(f"  - {department}: {', '.join(values)}")

    print(f"Output: {OUTPUT_PATH.resolve()}")


def main() -> None:
    records = build_records()
    if len(records) != 20:
        raise ValueError(f"Expected 20 total records, found {len(records)}")
    write_jsonl(records, OUTPUT_PATH)
    summarize(records)


if __name__ == "__main__":
    main()
