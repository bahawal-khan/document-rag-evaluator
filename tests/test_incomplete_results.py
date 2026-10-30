from types import SimpleNamespace as NS

from eval.common import summarize_results


def _md(name, score=1.0, success=True, error=None, reason=None):
    return NS(name=name, score=score, success=success, error=error, reason=reason)


def test_errored_metric_is_not_counted_as_zero_score():
    result = NS(test_results=[
        NS(input="q1", metrics_data=[_md("Faithfulness"), _md("Toxicity", error="429 quota")]),
        NS(input="q2", metrics_data=[_md("Faithfulness", 0.5, False, reason="bad")]),
    ])
    summary = summarize_results(result)
    assert summary["Toxicity"]["total"] == 0
    assert summary["Toxicity"]["errored"] == 1
    assert summary["Faithfulness"]["total"] == 2
    assert summary["Faithfulness"]["errored"] == 0
