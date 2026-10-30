from types import SimpleNamespace

from eval import suite
from eval.pipeline_eval import summarize_pipeline


def test_selected_evaluations_generate_answers_once_and_group_metrics(monkeypatch):
    calls = {"build": 0, "evaluate": 0}

    def fake_build_test_cases(
        golden_data,
        retriever,
        llm,
        prompt_builder,
        provider="groq",
        progress_callback=None,
    ):
        calls["build"] += 1
        assert golden_data[0]["input"] == "test question"
        assert retriever is not None and llm is not None and prompt_builder is not None
        assert provider == "groq"
        return ["test-case"]

    def fake_evaluate(test_cases, metrics):
        calls["evaluate"] += 1
        assert test_cases == ["test-case"]
        assert len(metrics) == 9
        metric_names = [
            "Contextual Precision",
            "Contextual Recall",
            "Contextual Relevancy",
            "Answer Relevancy",
            "Faithfulness",
            "Correctness",
            "Completeness",
            "Toxicity",
            "PII Leakage",
        ]
        metric_data = [
            SimpleNamespace(name=name, score=1.0, success=True, reason=None)
            for name in metric_names
        ]
        test_result = SimpleNamespace(
            metrics_data=metric_data,
            success=True,
            metadata={"difficulty": "easy"},
            input="test question",
        )
        return SimpleNamespace(test_results=[test_result])

    monkeypatch.setattr(suite, "build_test_cases", fake_build_test_cases)
    monkeypatch.setattr(suite, "evaluate_test_cases", fake_evaluate)

    def fake_component_metrics(judge, *args):
        assert judge is not None
        return [1, 2, 3, 4, 5]

    def fake_quality_metrics(judge):
        assert judge is not None
        return [6, 7]

    def fake_safety_metrics(judge, *args):
        assert judge is not None
        return [8, 9]

    monkeypatch.setattr(suite, "build_component_metrics", fake_component_metrics)
    monkeypatch.setattr(suite, "build_quality_metrics", fake_quality_metrics)
    monkeypatch.setattr(suite, "build_safety_content_metrics", fake_safety_metrics)

    result = suite.run_selected_evaluations(
        [{"input": "test question", "expected_output": "expected"}],
        retriever=object(),
        llm=object(),
        judge=object(),
        run_component=True,
        run_quality=True,
        run_safety=True,
    )

    assert calls == {"build": 1, "evaluate": 1}
    assert set(result["retriever_summary"]) == {
        "Contextual Precision",
        "Contextual Recall",
        "Contextual Relevancy",
    }
    assert set(result["generator_summary"]) == {"Answer Relevancy", "Faithfulness"}
    assert set(result["quality_summary"]) == {"Correctness", "Completeness"}
    assert set(result["safety_content_summary"]) == {"Toxicity", "PII Leakage"}
    assert result["pipeline_summary"]["total_questions"] == 1
    assert result["pipeline_summary"]["fully_passed"] == 1
    assert set(result["pipeline_summary"]["metric_summary"]) == {
        "Contextual Precision",
        "Contextual Recall",
        "Contextual Relevancy",
        "Answer Relevancy",
        "Faithfulness",
    }


def test_pipeline_difficulty_breakdown_uses_the_selected_metrics():
    component_pass_quality_fail = SimpleNamespace(
        name="Contextual Precision", success=True, score=1.0, reason=None
    )
    quality_fail = SimpleNamespace(
        name="Correctness", success=False, score=0.0, reason="Incorrect"
    )
    component_fail_quality_pass = SimpleNamespace(
        name="Contextual Precision", success=False, score=0.0, reason="Not precise"
    )
    quality_pass = SimpleNamespace(
        name="Correctness", success=True, score=1.0, reason=None
    )
    results = SimpleNamespace(
        test_results=[
            SimpleNamespace(
                metrics_data=[component_pass_quality_fail, quality_fail],
                success=False,
                metadata={"difficulty": "easy"},
                input="easy question",
            ),
            SimpleNamespace(
                metrics_data=[component_fail_quality_pass, quality_pass],
                success=False,
                metadata={"difficulty": "hard"},
                input="hard question",
            ),
        ]
    )

    summary = summarize_pipeline(results, metric_names=["Contextual Precision"])

    assert summary["fully_passed"] == 1
    assert summary["difficulty_breakdown"]["easy"] == {
        "passed": 1,
        "total": 1,
        "pass_rate": 1.0,
    }
    assert summary["difficulty_breakdown"]["hard"] == {
        "passed": 0,
        "total": 1,
        "pass_rate": 0.0,
    }


def test_pdf_report_escapes_uploaded_and_model_generated_markup():
    from eval.report_pdf import build_pdf_report

    untrusted_text = "<unexpected-tag> & untrusted text"
    metric_summary = {
        "Correctness": {
            "average": 0.0,
            "pass_rate": 0.0,
            "passed": 0,
            "total": 1,
            "failure_reasons": [{
                "input": untrusted_text,
                "score": 0.0,
                "reason": untrusted_text,
            }],
        }
    }
    pipeline_summary = {
        "metric_summary": {},
        "fully_passed": 0,
        "total_questions": 0,
        "fully_passed_rate": 0.0,
        "difficulty_breakdown": {},
    }

    report = build_pdf_report(
        metric_summary,
        {},
        pipeline_summary,
        {},
        {},
        {},
        None,
    )

    assert report.startswith(b"%PDF")
