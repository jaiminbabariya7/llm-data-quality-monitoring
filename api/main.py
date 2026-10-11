"""FastAPI application for the LLM-Powered Data Quality Monitor."""
from __future__ import annotations
import logging, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import MonitorRequest, MonitorResponse, AnomalyDetail, HealthResponse
from src.bigquery_connector import BigQueryConnector
from src.data_profiler import DataProfiler, load_profile, save_profile
from src.anomaly_detector import AnomalyDetector
from src.llm_explainer import LLMExplainer
from src.alert_manager import AlertManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Data Quality Monitor",
    description="BigQuery data quality monitoring with OpenAI explanations and a rule-based fallback",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health check."""
    return HealthResponse(status="healthy")


@app.post("/monitor", response_model=MonitorResponse)
def run_monitor(req: MonitorRequest):
    """Run a full data quality check on a BigQuery table.

    Profiles the table, detects anomalies, generates an LLM explanation,
    and sends alerts to configured channels.
    """
    try:
        bq        = BigQueryConnector(project_id=req.project_id)
        profiler  = DataProfiler(connector=bq)
        detector  = AnomalyDetector(connector=bq)
        detector.NULL_RATE_WARNING = req.null_rate_threshold
        explainer = LLMExplainer()
        alerter   = AlertManager(bq_connector=bq)

        # 1. Profile table
        profile = profiler.profile_table(req.dataset, req.table)

        # 2. Detect anomalies against the previous profile, then store this one
        baseline = load_profile(req.dataset, req.table)
        report = detector.detect(profile, baseline=baseline,
                                 lookback_days=req.lookback_days, date_col=req.date_column)
        save_profile(profile)

        # 3. Generate LLM explanation
        explanation = explainer.explain(report)

        # 4. Send alerts
        alert_results = alerter.send_all(report, explanation) if report.has_anomalies else {}

        return MonitorResponse(
            table=f"{req.dataset}.{req.table}",
            run_id=report.run_id,
            anomalies_found=len(report.anomalies),
            critical_count=report.critical_count,
            anomalies=[AnomalyDetail(anomaly_type=a.anomaly_type, column=a.column,
                                     description=a.description, severity=a.severity)
                       for a in report.anomalies],
            llm_explanation=explanation,
            alert_channels=alert_results,
            profiled_at=profile.profiled_at,
        )
    except Exception as e:
        logger.exception("Monitor run failed for %s.%s", req.dataset, req.table)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{dataset}/{table}")
def get_report(dataset: str, table: str, limit: int = 10):
    """Fetch the N most recent quality-check reports for a table."""
    try:
        bq = BigQueryConnector()
        sql = f"""
            SELECT run_id, anomaly_count, critical_count, llm_explanation, created_at
            FROM `{bq.project_id}.{dataset}.dq_audit_log`
            WHERE dataset = '{dataset}' AND table_name = '{table}'
            ORDER BY created_at DESC LIMIT {limit}
        """
        df = bq.query_df(sql)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
