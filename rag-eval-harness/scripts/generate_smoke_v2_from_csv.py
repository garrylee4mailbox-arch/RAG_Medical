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


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "Chinese-medical-dialogue-data-master"
OUTPUT_PATH = REPO_ROOT / "data" / "questions" / "smoke_v2.0.jsonl"

# Fixed row indices keep regeneration deterministic and directly traceable to
# the original CSVs. The output uses the raw ask/answer fields with whitespace
# normalization only.
SELECTIONS: Dict[str, List[Selection]] = {
    "Oncology": [
        Selection(0),
        Selection(1),
        Selection(22),
        Selection(35),
        Selection(44),
        Selection(62),
        Selection(80),
        Selection(88),
        Selection(97),
        Selection(184),
    ],
    "Pediatric": [
        Selection(0),
        Selection(74),
        Selection(188),
        Selection(194),
        Selection(195),
        Selection(197),
        Selection(198),
        Selection(199),
        Selection(201),
        Selection(202),
    ],
}


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
    records: List[Dict[str, object]] = []
    seen_sources = set()

    for source_group, selections in SELECTIONS.items():
        source_csv = find_source_csv(source_group)
        rows = load_rows(source_csv)

        for ordinal, selection in enumerate(selections, start=1):
            if selection.row_index >= len(rows):
                raise IndexError(f"Row {selection.row_index} is out of range for {source_csv}")
            source_key = (source_group, selection.row_index)
            if source_key in seen_sources:
                raise ValueError(f"Duplicate source row selected: {source_group} row {selection.row_index}")
            seen_sources.add(source_key)
            records.append(
                make_record(
                    source_csv=source_csv,
                    row=rows[selection.row_index],
                    selection=selection,
                    source_group=source_group,
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
    for source_group in SELECTIONS:
        first_record = next(record for record in records if record["source_group"] == normalize_department(source_group))
        print(f"  - {first_record['source_csv']}")

    print("Selection counts:")
    for source_group in SELECTIONS:
        count = sum(1 for record in records if record["source_group"] == normalize_department(source_group))
        print(f"  - {source_group}: {count}")

    print("Diseases:")
    for source_group in SELECTIONS:
        values = sorted({str(record["disease"]) for record in records if record["source_group"] == normalize_department(source_group)})
        print(f"  - {source_group}: {', '.join(values)}")

    print(f"Output: {OUTPUT_PATH.resolve()}")


def main() -> None:
    records = build_records()
    if len(records) != 20:
        raise ValueError(f"Expected 20 total records, found {len(records)}")
    write_jsonl(records, OUTPUT_PATH)
    summarize(records)


if __name__ == "__main__":
    main()
