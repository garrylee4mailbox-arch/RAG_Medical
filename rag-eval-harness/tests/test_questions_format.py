import json
from pathlib import Path

def _check(path: Path):
    assert path.exists()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        assert obj.get("bucket")
        assert obj.get("question")

def test_smoke_questions():
    _check(Path(__file__).resolve().parents[1] / "data/questions/smoke_v1.jsonl")

def test_full_questions():
    _check(Path(__file__).resolve().parents[1] / "data/questions/full_v1.jsonl")
