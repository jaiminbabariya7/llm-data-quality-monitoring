"""
Airflow DAG: Daily Data Quality Monitoring Pipeline.

Runs profiling + anomaly detection + LLM explanation on all
registered BigQuery tables and sends alerts on any findings.

Schedule: Daily at 06:30 UTC (after main ETL pipelines finish).
"""
from __future__ import annotations
import json, logging
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# Tables to monitor — extend this list as your data grows
MONITORED_TABLES = [
    {"dataset": "ecommerce", "table": "orders"},
    {"dataset": "ecommerce", "table": "customers"},
    {"dataset": "ecommerce", "table": "products"},
    {"dataset": "marketing", "table": "campaigns"},
]

DEFAULT_ARGS = {
    "owner": "jaimin.babariya",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}


def _run_quality_check(dataset: str, table: str, **kwargs) -> dict:
    """Profile table, detect anomalies, and return results to XCom."""
    import os, sys
    sys.path.insert(0, "/opt/airflow")
    from src.bigquery_connector import BigQueryConnector
    from src.data_profiler import DataProfiler, load_profile, save_profile
    from src.anomaly_detector import AnomalyDetector
    from src.llm_explainer import LLMExplainer
    from src.alert_manager import AlertManager

    bq        = BigQueryConnector(project_id=os.getenv("GCP_PROJECT_ID"))
    profile   = DataProfiler(connector=bq).profile_table(dataset, table)
    baseline  = load_profile(dataset, table)          # None on first run
    report    = AnomalyDetector(connector=bq).detect(profile, baseline=baseline)
    save_profile(profile)
    explanation = LLMExplainer().explain(report)

    if report.has_anomalies:
        AlertManager(bq_connector=bq).send_all(report, explanation)
        logger.warning("[%s.%s] %d anomalies found (%d critical)",
                       dataset, table, len(report.anomalies), report.critical_count)
    else:
        logger.info("[%s.%s] All quality checks passed", dataset, table)

    return {"table": f"{dataset}.{table}", "anomaly_count": len(report.anomalies),
            "critical": report.critical_count, "run_id": report.run_id}


def _check_results(**kwargs) -> str:
    """Branch: fail the DAG if any critical anomalies were found."""
    ti = kwargs["ti"]
    all_critical = sum(
        ti.xcom_pull(task_ids=f"check_{t['dataset']}_{t['table']}").get("critical", 0)
        for t in MONITORED_TABLES
    )
    return "mark_failed" if all_critical > 0 else "mark_passed"


with DAG(
    dag_id="data_quality_monitor",
    description="Daily LLM-powered data quality checks across all BigQuery tables",
    default_args=DEFAULT_ARGS,
    schedule_interval="30 6 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["data-quality","llm","bigquery","monitoring"],
) as dag:

    start = EmptyOperator(task_id="start")

    quality_tasks = [
        PythonOperator(
            task_id=f"check_{t['dataset']}_{t['table']}",
            python_callable=_run_quality_check,
            op_kwargs={"dataset": t["dataset"], "table": t["table"]},
        )
        for t in MONITORED_TABLES
    ]

    branch = BranchPythonOperator(task_id="evaluate_results",
                                   python_callable=_check_results)
    mark_passed = EmptyOperator(task_id="mark_passed")
    mark_failed = EmptyOperator(task_id="mark_failed")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> quality_tasks >> branch >> [mark_passed, mark_failed] >> end
