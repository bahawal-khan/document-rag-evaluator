"""
COMPONENT-LEVEL evaluation.

Retriever metrics  — Contextual Precision, Contextual Recall, Contextual Relevancy
Generator metrics  — Answer Relevancy, Faithfulness

(Technically these run within a single evaluate() call, since each metric
makes its own LLM calls regardless of how they're grouped — the grouping
below is purely for REPORTING, splitting results into "retriever" vs
"generator" sections.)
"""

from core.config import METRIC_THRESHOLD
from core.generator import build_prompt_within_budget
from eval.common import build_test_cases, evaluate_test_cases, summarize_results
from deepeval.metrics import (
    ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric,
    AnswerRelevancyMetric, FaithfulnessMetric,
)

RETRIEVER_METRICS = ["Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
GENERATOR_METRICS = ["Answer Relevancy", "Faithfulness"]


def build_component_metrics(judge, include_reasons: bool = True):
    return [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, include_reason=include_reasons, model=judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, include_reason=include_reasons, model=judge),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, include_reason=include_reasons, model=judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, include_reason=include_reasons, model=judge),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, include_reason=include_reasons, model=judge),
    ]


def run_component_eval(golden_data, retriever, llm, judge, test_cases=None, include_reasons: bool = True):
    if test_cases is None:
        test_cases = build_test_cases(golden_data, retriever, llm, build_prompt_within_budget)

    result = evaluate_test_cases(test_cases, build_component_metrics(judge, include_reasons))

    summary = summarize_results(result)
    retriever_summary = {k: v for k, v in summary.items() if k in RETRIEVER_METRICS}
    generator_summary = {k: v for k, v in summary.items() if k in GENERATOR_METRICS}
    return retriever_summary, generator_summary, result
