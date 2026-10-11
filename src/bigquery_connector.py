"""BigQuery client wrapper with retry logic and schema introspection."""
from __future__ import annotations
import logging, os
from typing import Any
import pandas as pd
from google.cloud import bigquery
from google.api_core.retry import Retry

logger = logging.getLogger(__name__)
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project")

class BigQueryConnector:
    """Thin wrapper around the BigQuery client."""

    def __init__(self, project_id: str = PROJECT_ID) -> None:
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id

    def query_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return result as a DataFrame."""
        logger.debug("Executing query: %s", sql[:120])
        return self.client.query(sql, retry=Retry()).to_dataframe()

    def get_table_schema(self, dataset: str, table: str) -> list[dict]:
        """Return list of {name, field_type, mode} dicts for each column."""
        ref = self.client.get_table(f"{self.project_id}.{dataset}.{table}")
        return [{"name": f.name, "type": f.field_type, "mode": f.mode}
                for f in ref.schema]

    def get_row_count(self, dataset: str, table: str) -> int:
        """Return approximate row count from table metadata."""
        ref = self.client.get_table(f"{self.project_id}.{dataset}.{table}")
        return ref.num_rows

    def table_sample(self, dataset: str, table: str,
                     limit: int = 100_000) -> pd.DataFrame:
        """Return a random sample of up to `limit` rows."""
        sql = (f"SELECT * FROM `{self.project_id}.{dataset}.{table}`"
               f" TABLESAMPLE SYSTEM (10 PERCENT) LIMIT {limit}")
        return self.query_df(sql)

    def get_daily_row_counts(self, dataset: str, table: str,
                              date_col: str, lookback_days: int = 30) -> pd.DataFrame:
        """Return daily row counts over the last N days."""
        sql = f"""
            SELECT DATE({date_col}) AS dt, COUNT(*) AS row_count
            FROM `{self.project_id}.{dataset}.{table}`
            WHERE DATE({date_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
            GROUP BY dt ORDER BY dt
        """
        return self.query_df(sql)

    def write_audit_log(self, dataset: str, records: list[dict]) -> None:
        """Append quality-check results to the dq_audit_log table."""
        table_id = f"{self.project_id}.{dataset}.dq_audit_log"
        errors = self.client.insert_rows_json(table_id, records)
        if errors:
            logger.error("BigQuery insert errors: %s", errors)
        else:
            logger.info("Wrote %d audit records to %s", len(records), table_id)
