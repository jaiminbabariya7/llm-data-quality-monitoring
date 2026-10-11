"""
Automated statistical profiling of BigQuery tables.

Computes null rates, cardinality, numeric distributions, and
schema snapshots — the foundation for anomaly detection.
"""
from __future__ import annotations
import json, logging, os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .bigquery_connector import BigQueryConnector

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_rate: float
    cardinality: int
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    p25: float | None = None
    p75: float | None = None
    top_values: list[Any] = field(default_factory=list)


@dataclass
class TableProfile:
    dataset: str
    table: str
    row_count: int
    schema: list[dict]
    columns: list[ColumnProfile]
    profiled_at: str = ""


# Previous profiles are kept as JSON so each run can compare against the last
# one (mean drift and schema changes need a baseline).
PROFILE_STORE_DIR = os.getenv("PROFILE_STORE_DIR", "profiles")


def _profile_path(dataset: str, table: str, store_dir: str | None = None) -> Path:
    return Path(store_dir or PROFILE_STORE_DIR) / f"{dataset}.{table}.json"


def save_profile(profile: TableProfile, store_dir: str | None = None) -> Path:
    """Persist a profile as the baseline for the next run."""
    path = _profile_path(profile.dataset, profile.table, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), default=str, indent=2))
    return path


def load_profile(dataset: str, table: str,
                 store_dir: str | None = None) -> TableProfile | None:
    """Load the previous profile for a table, or None on the first run."""
    path = _profile_path(dataset, table, store_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["columns"] = [ColumnProfile(**c) for c in data["columns"]]
    return TableProfile(**data)


class DataProfiler:
    """Profile BigQuery tables and return structured statistics."""

    NULL_RATE_THRESHOLD = 0.05   # 5% null → warning
    LOW_CARDINALITY_THRESHOLD = 5

    def __init__(self, connector: BigQueryConnector | None = None) -> None:
        self.bq = connector or BigQueryConnector()

    def profile_table(self, dataset: str, table: str,
                      sample_rows: int = 100_000) -> TableProfile:
        """Run full profiling pass on a BigQuery table.

        Args:
            dataset: BigQuery dataset name.
            table: Table name.
            sample_rows: Max rows to sample for column statistics.

        Returns:
            TableProfile with per-column statistics.
        """
        from datetime import datetime, timezone
        logger.info("Profiling %s.%s ...", dataset, table)
        schema = self.bq.get_table_schema(dataset, table)
        row_count = self.bq.get_row_count(dataset, table)
        df = self.bq.table_sample(dataset, table, limit=sample_rows)

        columns = [self._profile_column(df, col_meta) for col_meta in schema
                   if col_meta["name"] in df.columns]

        return TableProfile(
            dataset=dataset, table=table, row_count=row_count,
            schema=schema, columns=columns,
            profiled_at=datetime.now(timezone.utc).isoformat(),
        )

    def _profile_column(self, df: pd.DataFrame,
                        col_meta: dict) -> ColumnProfile:
        """Compute statistics for a single column."""
        name = col_meta["name"]
        series = df[name]
        null_rate = series.isna().mean()
        cardinality = series.nunique()

        prof = ColumnProfile(
            name=name, dtype=col_meta["type"],
            null_rate=round(float(null_rate), 4),
            cardinality=int(cardinality),
            top_values=series.dropna().value_counts().head(5).index.tolist(),
        )

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean):
                prof.mean    = round(float(clean.mean()), 4)
                prof.std     = round(float(clean.std()),  4)
                prof.min_val = round(float(clean.min()),  4)
                prof.max_val = round(float(clean.max()),  4)
                prof.p25     = round(float(clean.quantile(0.25)), 4)
                prof.p75     = round(float(clean.quantile(0.75)), 4)

        return prof

    def flag_null_issues(self, profile: TableProfile) -> list[str]:
        """Return column names whose null rate exceeds threshold."""
        return [
            f"{c.name}: {c.null_rate*100:.1f}% null (threshold {self.NULL_RATE_THRESHOLD*100:.0f}%)"
            for c in profile.columns if c.null_rate > self.NULL_RATE_THRESHOLD
        ]
