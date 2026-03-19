from src.eval.io_utils import aggregate_metrics
from src.eval.schemas import AnswerRecord


def test_aggregate_metrics_includes_answer_similarity_columns():
    records = [
        AnswerRecord(
            run_id="run1",
            question_id="q1",
            bucket="bucket1",
            question="question",
            system_name="rag",
            answer="answer",
            gold_answer="gold",
            answer_similarity=0.75,
            normalized_answer_for_scoring="answer",
            normalized_gold_answer_for_scoring="gold",
            contexts=[],
            meta={"retrieval_count": 1},
        )
    ]

    metrics = aggregate_metrics(records)
    assert metrics
    row = metrics[0]
    assert "answer_similarity_avg" in row
    assert "answer_scored_rate" in row
    assert "answer_similarity_std" in row
    assert "answer_similarity_min" in row
    assert "answer_similarity_max" in row
