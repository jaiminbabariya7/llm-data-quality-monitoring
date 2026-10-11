"""
Statistical anomaly detection for BigQuery table profiles.

Applies null-rate, Z-score, IQR outlier, mean drift, volume drift, and
schema-change detection
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
    anomaly_type: str       # "null_rate" | "zscore" | "iqr_outlier" | "mean_drift" | "volume_drift" | "schema_change"
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

    ZSCORE_THRESHOLD   = 3.0    # |z| above this → outlier / drift
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
               lookback_days: int = 7,
               date_col: str = "created_at") -> AnomalyReport:
        """Run all detectors and return a consolidated AnomalyReport.

        Args:
            profile: Current table profile.
            baseline: Previous profile for schema-change comparison (optional).
            lookback_days: Days of history for volume drift checks.
            date_col: Date/timestamp column used to count rows per day.

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

        # 2. Numeric Z-score + IQR outlier fence
        anomalies.extend(self._check_distributions(profile))

        # 3. Mean drift vs. previous profile (Z-score against baseline)
        if baseline:
            anomalies.extend(self._check_mean_drift(profile, baseline))

        # 4. Volume drift (row count trend)
        anomalies.extend(self._check_volume_drift(
            profile.dataset, profile.table, lookback_days, date_col))

        # 5. Schema changes
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
            # Z-score of the column extremes: how many standard deviations
            # the min / max sit from the mean.
            extremes = [v for v in (col.min_val, col.max_val) if v is not None]
            if extremes:
                worst = max(extremes, key=lambda v: abs(v - col.mean))
                z = (worst - col.mean) / col.std
                if abs(z) > self.ZSCORE_THRESHOLD:
                    result.append(Anomaly(
                        anomaly_type="zscore", column=col.name,
                        description=f"{col.name}: extreme value {worst:.2f} is {abs(z):.1f} standard deviations from the mean {col.mean:.2f}",
                        severity="medium", metric_value=round(z, 2), threshold=self.ZSCORE_THRESHOLD))
            if col.p75 is not None and col.p25 is not None:
                iqr = col.p75 - col.p25
                lower, upper = col.p25 - 1.5 * iqr, col.p75 + 1.5 * iqr
                if col.max_val > upper or col.min_val < lower:
                    result.append(Anomaly(
                        anomaly_type="iqr_outlier", column=col.name,
                        description=f"{col.name}: values outside IQR fence [{lower:.2f}, {upper:.2f}] — possible outliers",
                        severity="low", metric_value=col.max_val, threshold=upper))
        return result

    def _check_mean_drift(self, profile: TableProfile,
                          baseline: TableProfile) -> list[Anomaly]:
        """Flag numeric columns whose mean moved more than ZSCORE_THRESHOLD
        baseline standard deviations since the previous profile."""
        prev = {c.name: c for c in baseline.columns}
        result = []
        for col in profile.columns:
            base = prev.get(col.name)
            if (col.mean is None or base is None or base.mean is None
                    or base.std is None or base.std == 0):
                continue
            z = (col.mean - base.mean) / base.std
            if abs(z) > self.ZSCORE_THRESHOLD:
                result.append(Anomaly(
                    anomaly_type="mean_drift", column=col.name,
                    description=f"{col.name}: mean moved from {base.mean:.2f} to {col.mean:.2f} ({z:+.1f} baseline standard deviations)",
                    severity="high", metric_value=round(z, 2), threshold=self.ZSCORE_THRESHOLD))
        return result

    def _check_volume_drift(self, dataset: str, table: str,
                            lookback_days: int, date_col: str) -> list[Anomaly]:
        """Compare the latest day's row count with the average of the previous days."""
        try:
            df = self.bq.get_daily_row_counts(dataset, table, date_col=date_col,
                                              lookback_days=lookback_days)
        except Exception as e:
            logger.warning("Volume drift check skipped: %s", e)
            return []
        return self.volume_drift_from_counts(df, lookback_days)

    def volume_drift_from_counts(self, df: pd.DataFrame, lookback_days: int) -> list[Anomaly]:
        """df: one row per day (dt, row_count), oldest first."""
        if df is None or len(df) < 4:                  # latest day + at least 3 history days
            return []
        df = df.sort_values("dt")
        latest = int(df["row_count"].iloc[-1])
        avg = float(df["row_count"].iloc[:-1].mean())
        if avg == 0:
            return []
        ratio = latest / avg
        day = df["dt"].iloc[-1]
        if ratio < (1 - self.VOLUME_DROP_PCT):
            return [Anomaly(
                anomaly_type="volume_drift", column=None,
                description=f"{day}: {latest:,} rows is {(1-ratio)*100:.0f}% below the previous {len(df)-1}-day average {avg:,.0f}",
                severity="critical", metric_value=round(ratio, 3), threshold=1-self.VOLUME_DROP_PCT)]
        if ratio > self.VOLUME_SPIKE_PCT:
            return [Anomaly(
                anomaly_type="volume_drift", column=None,
                description=f"{day}: {latest:,} rows is {ratio:.1f}x the previous {len(df)-1}-day average — possible duplication",
                severity="high", metric_value=round(ratio, 3), threshold=self.VOLUME_SPIKE_PCT)]
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
