"""
Statistical anomaly detection for BigQuery table profiles.

Applies null-rate, IQR outlier, volume drift, and schema-change detection
to flag data quality issues before they reach downstream consumers.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from .data_profiler import TableProfile, ColumnProfile
from .bigquery_connector import BigQueryConnector

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    anomaly_type: str       # "null_rate" | "iqr_outlier" | "volume_drift" | "schema_change"
    column: str | None
    description: str
    severity: str           # "low" | "medium" | "high" | "critical"
    metric_value: float | None = None
    threshold: float | None = None


@dataclass
class AnomalyReport:
    dataset: str
    table: str
    run_id: str
    anomalies: list[Anomaly] = field(default_factory=list)
    previous_schema: list[dict] = field(default_factory=list)

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.anomalies if a.severity == "critical")


class AnomalyDetector:
    """Detect statistical anomalies in BigQuery table profiles."""

    VOLUME_DROP_PCT    = 0.30   # 30% drop → critical
    VOLUME_SPIKE_PCT   = 2.00   # 200% spike → high
    NULL_RATE_WARNING  = 0.05
    NULL_RATE_CRITICAL = 0.20

    def __init__(self, connector: BigQueryConnector | None = None) -> None:
        self.bq = connector or BigQueryConnector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, profile: TableProfile,
               baseline: TableProfile | None = None,
               lookback_days: int = 7) -> AnomalyReport:
        """Run all detectors and return a consolidated AnomalyReport.

        Args:
            profile: Current table profile.
            baseline: Previous profile for schema-change comparison (optional).
            lookback_days: Days of history for volume drift checks.

        Returns:
            AnomalyReport with all detected anomalies.
        """
        from datetime import datetime, timezone
        run_id = f"qc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        report = AnomalyReport(dataset=profile.dataset, table=profile.table,
                               run_id=run_id,
                               previous_schema=baseline.schema if baseline else [])
        anomalies: list[Anomaly] = []

        # 1. Null rate checks
        anomalies.extend(self._check_null_rates(profile))

        # 2. Numeric IQR outlier fence
        anomalies.extend(self._check_distributions(profile))

        # 3. Volume drift (row count trend)
        anomalies.extend(self._check_volume_drift(
            profile.dataset, profile.table, profile.row_count, lookback_days))

        # 4. Schema changes
        if baseline:
            anomalies.extend(self._check_schema_changes(profile.schema, baseline.schema))

        report.anomalies = anomalies
        logger.info("Detected %d anomalies in %s.%s",
                    len(anomalies), profile.dataset, profile.table)
        return report

    # ------------------------------------------------------------------
    # Private detectors
    # ------------------------------------------------------------------

    def _check_null_rates(self, profile: TableProfile) -> list[Anomaly]:
        result = []
        for col in profile.columns:
            if col.null_rate >= self.NULL_RATE_CRITICAL:
                result.append(Anomaly(
                    anomaly_type="null_rate", column=col.name,
                    description=f"{col.name}: {col.null_rate*100:.1f}% null — critical threshold {self.NULL_RATE_CRITICAL*100:.0f}%",
                    severity="critical", metric_value=col.null_rate,
                    threshold=self.NULL_RATE_CRITICAL))
            elif col.null_rate >= self.NULL_RATE_WARNING:
                result.append(Anomaly(
                    anomaly_type="null_rate", column=col.name,
                    description=f"{col.name}: {col.null_rate*100:.1f}% null — above warning threshold {self.NULL_RATE_WARNING*100:.0f}%",
                    severity="medium", metric_value=col.null_rate,
                    threshold=self.NULL_RATE_WARNING))
        return result

    def _check_distributions(self, profile: TableProfile) -> list[Anomaly]:
        result = []
        for col in profile.columns:
            if col.mean is None or col.std is None or col.std == 0:
                continue
            if col.p75 is not None and col.p25 is not None:
                iqr = col.p75 - col.p25
                lower, upper = col.p25 - 1.5 * iqr, col.p75 + 1.5 * iqr
                if col.max_val > upper or col.min_val < lower:
                    result.append(Anomaly(
                        anomaly_type="iqr_outlier", column=col.name,
                        description=f"{col.name}: values outside IQR fence [{lower:.2f}, {upper:.2f}] — possible outliers",
                        severity="low", metric_value=col.max_val, threshold=upper))
        return result

    def _check_volume_drift(self, dataset: str, table: str,
                             current_rows: int, lookback_days: int) -> list[Anomaly]:
        try:
            df = self.bq.get_daily_row_counts(dataset, table,
                                               date_col="created_at",
                                               lookback_days=lookback_days)
            if len(df) < 3:
                return []
            avg = df["row_count"].mean()
            if avg == 0:
                return []
            ratio = current_rows / avg
            if ratio < (1 - self.VOLUME_DROP_PCT):
                return [Anomaly(
                    anomaly_type="volume_drift", column=None,
                    description=f"Row count {current_rows:,} is {(1-ratio)*100:.0f}% below {lookback_days}-day average {avg:,.0f}",
                    severity="critical", metric_value=ratio, threshold=1-self.VOLUME_DROP_PCT)]
            if ratio > self.VOLUME_SPIKE_PCT:
                return [Anomaly(
                    anomaly_type="volume_drift", column=None,
                    description=f"Row count {current_rows:,} is {ratio:.1f}x above {lookback_days}-day average — possible data duplication",
                    severity="high", metric_value=ratio, threshold=self.VOLUME_SPIKE_PCT)]
        except Exception as e:
            logger.warning("Volume drift check skipped: %s", e)
        return []

    def _check_schema_changes(self, current: list[dict],
                               previous: list[dict]) -> list[Anomaly]:
        curr_names = {c["name"] for c in current}
        prev_names = {c["name"] for c in previous}
        result = []
        for added in curr_names - prev_names:
            result.append(Anomaly(
                anomaly_type="schema_change", column=added,
                description=f"New column '{added}' added since last run — verify downstream models",
                severity="medium"))
        for removed in prev_names - curr_names:
            result.append(Anomaly(
                anomaly_type="schema_change", column=removed,
                description=f"Column '{removed}' removed — downstream queries may break",
                severity="critical"))
        return result
