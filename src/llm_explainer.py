"""
LLM-based anomaly explanation and remediation suggestions.

Calls an OpenAI model (gpt-4o-mini by default, set via LLM_MODEL), with a rule-based fallback, to generate
plain-English summaries of data quality anomalies and actionable
remediation steps for the data engineering team.
"""
from __future__ import annotations
import logging, os
from .anomaly_detector import AnomalyReport, Anomaly

logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _build_prompt(report: AnomalyReport) -> str:
    lines = [
        f"You are a senior data engineer reviewing automated data quality alerts.",
        f"Table: {report.dataset}.{report.table}  |  Run ID: {report.run_id}",
        f"Anomalies detected ({len(report.anomalies)}):",
    ]
    for i, a in enumerate(report.anomalies, 1):
        lines.append(f"  {i}. [{a.severity.upper()}] {a.description}")
    lines += [
        "",
        "Please provide:",
        "1. A concise plain-English summary of what these anomalies mean.",
        "2. The most likely root causes for each issue.",
        "3. Specific, actionable remediation steps the team should take.",
        "Keep the response focused and practical — no more than 200 words.",
    ]
    return "
".join(lines)


def explain_with_openai(report: AnomalyReport) -> str:
    """Call the configured OpenAI model to explain anomalies in plain English.

    Args:
        report: AnomalyReport containing detected anomalies.

    Returns:
        Plain-English explanation string.

    Raises:
        RuntimeError: If OpenAI call fails and fallback is unavailable.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _build_prompt(report)}],
            max_tokens=400,
            temperature=0.3,
        )
        explanation = resp.choices[0].message.content.strip()
        logger.info("LLM explanation generated (%d chars)", len(explanation))
        return explanation
    except Exception as e:
        logger.warning("OpenAI call failed (%s) — using rule-based fallback", e)
        return _rule_based_fallback(report)


def _rule_based_fallback(report: AnomalyReport) -> str:
    """Deterministic fallback when OpenAI is unavailable."""
    lines = [f"Data quality report for {report.dataset}.{report.table}:", ""]
    critical = [a for a in report.anomalies if a.severity == "critical"]
    others   = [a for a in report.anomalies if a.severity != "critical"]
    if critical:
        lines.append(f"CRITICAL ({len(critical)} issues require immediate attention):")
        for a in critical:
            lines.append(f"  - {a.description}")
    if others:
        lines.append(f"Warnings ({len(others)} issues to review):")
        for a in others:
            lines.append(f"  - {a.description}")
    lines += ["", "Recommended actions:",
              "1. Review the upstream pipeline that populates this table.",
              "2. Check for ETL job failures in the last 24 hours.",
              "3. Verify schema changes are reflected in downstream models."]
    return "
".join(lines)


class LLMExplainer:
    """Wrapper that selects the best available LLM backend."""

    def explain(self, report: AnomalyReport) -> str:
        if not report.has_anomalies:
            return f"No anomalies detected in {report.dataset}.{report.table}. All checks passed."
        if OPENAI_API_KEY:
            return explain_with_openai(report)
        return _rule_based_fallback(report)
