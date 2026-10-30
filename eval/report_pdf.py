"""
Consolidates every eval result (component, pipeline, application
quality/safety, operational) into a single downloadable PDF report.
"""

import io
from datetime import datetime
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from core.config import P50_TARGET_SECONDS, P50_HARD_LIMIT_SECONDS, P99_HARD_LIMIT_SECONDS, METRIC_THRESHOLD

PASS_COLOR = colors.HexColor("#d4edda")
FAIL_COLOR = colors.HexColor("#f8d7da")
HEADER_COLOR = colors.HexColor("#343a40")
NO_DATA_COLOR = colors.HexColor("#e2e3e5")


def _pdf_text(value) -> str:
    """Escape untrusted text before placing it inside ReportLab Paragraph markup."""
    text = str(value or "")
    text = "".join(character for character in text if character in "\t\n\r" or ord(character) >= 32)
    return escape(text)


def _estimated_cost(value, places=5):
    return f"${value:.{places}f} est." if value is not None else "—"


def _metric_table(data: dict, title: str, styles):
    """data: {metric_name: {average, pass_rate, passed, failed, total}}"""
    rows = [["Metric", "Avg Score", "Pass Rate", "Passed / Total"]]
    row_colors = []
    for name, v in data.items():
        if name.startswith("_"):
            continue
        passed_ok = v["average"] >= METRIC_THRESHOLD
        rows.append([name, f"{v['average']:.2f}", f"{v['pass_rate']*100:.0f}%", f"{v['passed']}/{v['total']}"])
        row_colors.append(PASS_COLOR if passed_ok else FAIL_COLOR)

    table = Table(rows, colWidths=[6.5 * cm, 3 * cm, 3 * cm, 3.5 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    for i, c in enumerate(row_colors, start=1):
        style.append(("BACKGROUND", (0, i), (-1, i), c))
    table.setStyle(TableStyle(style))

    elements = [Paragraph(_pdf_text(title), styles["Heading2"]), Spacer(1, 6), table, Spacer(1, 14)]
    return elements


def _failure_reasons_section(all_summaries: dict, styles):
    """Appendix: a few real judge explanations for the lowest-scoring
    metrics, so the report shows WHY something scored low, not just the number."""
    elements = [Paragraph("Appendix: Why Metrics Failed", styles["Heading1"])]

    worst = []
    for level_name, summary in all_summaries.items():
        for metric_name, v in summary.items():
            if metric_name.startswith("_") or not v.get("failure_reasons"):
                continue
            worst.append((v["average"], level_name, metric_name, v["failure_reasons"]))
    worst.sort(key=lambda x: x[0])  # lowest average first

    if not worst:
        elements.append(Paragraph("No failure reasons recorded (either everything passed, or reasons weren't captured).", styles["Normal"]))
        return elements

    for avg, level_name, metric_name, reasons in worst[:5]:  # 5 worst metrics
        elements.append(Paragraph(f"{_pdf_text(metric_name)} ({_pdf_text(level_name)}) — avg {avg:.2f}", styles["Heading3"]))
        for r in reasons[:3]:  # 3 example reasons per metric
            q = _pdf_text((r.get("input") or "")[:100])
            reason = _pdf_text(r.get("reason", ""))
            elements.append(Paragraph(f"<b>Q:</b> {q}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Why it failed (score {r['score']:.2f}):</b> {reason}", styles["Normal"]))
            elements.append(Spacer(1, 6))
        elements.append(Spacer(1, 10))

    return elements


def build_pdf_report(
    retriever_summary, generator_summary, pipeline_summary,
    quality_summary, safety_content_summary, scope_summary,
    operational_metrics,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    # ---- Title ----
    story.append(Paragraph("RAG Evaluation Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 16))

    # ---- Executive summary ----
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    exec_rows = [["Category", "Status"]]
    exec_colors = []

    def _status(avg_scores):
        metric_scores = [
            value["average"]
            for name, value in avg_scores.items()
            if not name.startswith("_")
        ]
        if not metric_scores:
            return "NO DATA", NO_DATA_COLOR
        ok = all(score >= METRIC_THRESHOLD for score in metric_scores)
        return ("PASS", PASS_COLOR) if ok else ("NEEDS ATTENTION", FAIL_COLOR)

    for label, summ in [
        ("Retriever (Component)", retriever_summary),
        ("Generator (Component)", generator_summary),
        ("Pipeline (RAG Triad)", pipeline_summary["metric_summary"]),
        ("Application Quality", quality_summary),
        ("Application Safety", safety_content_summary),
        ("Scope Adherence", scope_summary),
    ]:
        status, color = _status(summ)
        exec_rows.append([label, status])
        exec_colors.append(color)

    if operational_metrics:
        p50_latency = operational_metrics.get("p50_latency_s")
        p99_latency = operational_metrics.get("p99_latency_s")
        op_ok = (
            p50_latency is not None
            and p99_latency is not None
            and p50_latency <= P50_HARD_LIMIT_SECONDS
            and p99_latency <= P99_HARD_LIMIT_SECONDS
        )
        exec_rows.append(["Operational (Latency)", "PASS" if op_ok else "NEEDS ATTENTION"])
        exec_colors.append(PASS_COLOR if op_ok else FAIL_COLOR)

    exec_table = Table(exec_rows, colWidths=[9 * cm, 6 * cm])
    exec_style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]
    for i, c in enumerate(exec_colors, start=1):
        exec_style.append(("BACKGROUND", (0, i), (-1, i), c))
    exec_table.setStyle(TableStyle(exec_style))
    story.append(exec_table)
    story.append(PageBreak())

    # ---- Component level ----
    story.append(Paragraph("1. Component-Level Evaluation", styles["Heading1"]))
    story.extend(_metric_table(retriever_summary, "Retriever Metrics", styles))
    story.extend(_metric_table(generator_summary, "Generator Metrics", styles))

    # ---- Pipeline level ----
    story.append(Paragraph("2. Pipeline-Level Evaluation (RAG Triad)", styles["Heading1"]))
    story.append(Paragraph(
        f"Questions fully passing ALL metrics together: "
        f"{pipeline_summary['fully_passed']}/{pipeline_summary['total_questions']} "
        f"({pipeline_summary['fully_passed_rate']*100:.0f}%)", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.extend(_metric_table(pipeline_summary["metric_summary"], "Combined Metrics", styles))

    db = pipeline_summary.get("difficulty_breakdown")
    if db:
        story.append(Paragraph("Pass Rate by Difficulty", styles["Heading2"]))
        diff_rows = [["Difficulty", "Passed / Total", "Pass Rate"]]
        diff_colors = []
        for level in ["easy", "medium", "hard"]:
            d = db.get(level, {})
            if not d.get("total"):
                diff_rows.append([level.capitalize(), "—", "—"])
                diff_colors.append(colors.white)
                continue
            rate = d["pass_rate"]
            diff_rows.append([level.capitalize(), f"{d['passed']}/{d['total']}", f"{rate*100:.0f}%"])
            diff_colors.append(PASS_COLOR if rate >= METRIC_THRESHOLD else FAIL_COLOR)
        diff_table = Table(diff_rows, colWidths=[5 * cm, 5 * cm, 5 * cm])
        diff_style = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]
        for i, c in enumerate(diff_colors, start=1):
            diff_style.append(("BACKGROUND", (0, i), (-1, i), c))
        diff_table.setStyle(TableStyle(diff_style))
        story.append(diff_table)
        story.append(Spacer(1, 14))

    # ---- Application level ----
    story.append(Paragraph("3. Application-Level Evaluation", styles["Heading1"]))
    story.extend(_metric_table(quality_summary, "Quality — Correctness & Completeness", styles))
    story.extend(_metric_table(safety_content_summary, "Safety — Toxicity & PII Leakage", styles))
    story.extend(_metric_table(scope_summary, "Safety — Scope Adherence", styles))
    fast_rate = scope_summary.get("_fast_check_catch_rate")
    if fast_rate is not None:
        story.append(Paragraph(
            f"The fast heuristic (retrieval-distance) scope filter caught {fast_rate*100:.0f}% "
            f"of off-topic probes before any LLM call (zero extra cost/latency for those).",
            styles["Normal"]))
        story.append(Spacer(1, 10))

    # ---- Operational level ----
    story.append(Paragraph("4. Operational Evaluation", styles["Heading1"]))
    if operational_metrics:
        om = operational_metrics
        p50_latency = om.get("p50_latency_s")
        p99_latency = om.get("p99_latency_s")
        p50_status = (
            "NO DATA" if p50_latency is None
            else "PASS" if p50_latency <= P50_HARD_LIMIT_SECONDS
            else "FAIL"
        )
        p99_status = (
            "NO DATA" if p99_latency is None
            else "PASS" if p99_latency <= P99_HARD_LIMIT_SECONDS
            else "FAIL"
        )
        op_rows = [
            ["Metric", "Value", "Target", "Status"],
            ["P50 Latency", f"{p50_latency:.2f}s" if p50_latency is not None else "—",
             f"≤ {P50_TARGET_SECONDS}s (hard limit {P50_HARD_LIMIT_SECONDS}s)",
             p50_status],
            ["P99 Latency", f"{p99_latency:.2f}s" if p99_latency is not None else "—",
             f"≤ {P99_HARD_LIMIT_SECONDS}s",
             p99_status],
            ["Generated Latency Samples", str(om.get("generation_requests", 0)), "—", "—"],
            ["Avg Cache Response", f"{om['avg_cache_latency_s']:.3f}s" if om.get("avg_cache_latency_s") is not None else "—", "—", "—"],
            ["Cache Hit Rate", f"{om['cache_hit_rate']*100:.0f}%", "—", "—"],
            ["Avg Estimated Cost / Request", _estimated_cost(om.get("avg_cost_usd")), "—", "—"],
            ["Total Estimated Cost", _estimated_cost(om.get("total_cost_usd"), 4), "—", "—"],
            ["Requests With Unpriced Models", str(om.get("cost_unknown_count", 0)), "—", "—"],
            ["Reliability", f"{om['reliability']*100:.1f}%", "—", "—"],
            ["Total Requests", str(om["total_requests"]), "—", "—"],
        ]
        op_colors = [None] * (len(op_rows) - 1)  # one entry per DATA row (header excluded)
        op_colors[0] = PASS_COLOR if op_rows[1][-1] == "PASS" else FAIL_COLOR   # P50 row
        op_colors[1] = PASS_COLOR if op_rows[2][-1] == "PASS" else FAIL_COLOR   # P99 row

        op_table = Table(op_rows, colWidths=[4.5 * cm, 3 * cm, 5 * cm, 2.5 * cm])
        op_style = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]
        for data_idx, c in enumerate(op_colors):
            if c:
                table_row = data_idx + 1  # +1 because row 0 is the header
                op_style.append(("BACKGROUND", (0, table_row), (-1, table_row), c))
        op_table.setStyle(TableStyle(op_style))
        story.append(op_table)
    else:
        story.append(Paragraph("No live-chat traffic recorded yet — chat with the assistant first, then regenerate this report.", styles["Normal"]))

    story.append(PageBreak())
    story.extend(_failure_reasons_section({
        "Retriever": retriever_summary,
        "Generator": generator_summary,
        "Quality": quality_summary,
        "Safety": safety_content_summary,
        "Scope": scope_summary,
    }, styles))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
