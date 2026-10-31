from types import SimpleNamespace as NS

from eval.pipeline_eval import summarize_pipeline
from eval.report_pdf import build_pdf_report

NAMES = ["Contextual Precision", "Contextual Recall", "Contextual Relevancy", "Answer Relevancy", "Faithfulness"]


def _md(name, success=True, error=None):
    return NS(name=name, success=success and not error, error=error, score=1.0 if success else 0.0, reason=None)


def _result(relevancy_ok=False, errored=None):
    metrics = [_md(n, success=(n != "Contextual Relevancy" or relevancy_ok), error=("429" if n == errored else None))
               for n in NAMES]
    return NS(input="q", metrics_data=metrics, metadata={"difficulty": "easy"}, success=False)


def test_one_weak_metric_gives_zero_strict_but_high_partial_credit():
    raw = NS(test_results=[_result() for _ in range(10)])
    summary = summarize_pipeline(raw, metric_names=NAMES)
    assert summary["fully_passed"] == 0
    partial = summary["partial"]
    assert partial["mostly_passed"] == 10
    assert partial["avg_metrics_passed"] == 4.0
    assert partial["incomplete_questions"] == 0


def test_unscored_metric_is_reported_as_incomplete_not_failed():
    raw = NS(test_results=[_result(relevancy_ok=True, errored="Faithfulness")])
    summary = summarize_pipeline(raw, metric_names=NAMES)
    assert summary["partial"]["incomplete_questions"] == 1
    assert summary["partial"]["avg_metrics_scored"] == 4.0


def test_pdf_builds_with_only_one_section_run():
    summary = {"Contextual Precision": {"average": 1.0, "pass_rate": 1.0, "passed": 1, "failed": 0, "total": 1}}
    pdf = build_pdf_report(summary, {}, None, None, None, None, None, run_notes=["note"])
    assert pdf.startswith(b"%PDF")


def test_pdf_builds_with_no_data_metric():
    summary = {"Toxicity": {"average": 0, "pass_rate": 0, "passed": 0, "failed": 0, "total": 0, "errored": 3}}
    assert build_pdf_report({}, {}, None, {}, summary, {}, None).startswith(b"%PDF")
