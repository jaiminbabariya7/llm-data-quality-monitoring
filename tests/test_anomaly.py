"""Unit tests for AnomalyDetector."""
import unittest
from unittest.mock import MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))

from src.anomaly_detector import AnomalyDetector, Anomaly
from src.data_profiler import ColumnProfile, TableProfile


def _make_profile(null_rates=None, row_count=10000):
    null_rates = null_rates or {}
    cols = []
    for name in ["order_id","amount","email"]:
        nr = null_rates.get(name, 0.0)
        cols.append(ColumnProfile(name=name, dtype="STRING" if name=="email" else "FLOAT64",
                                   null_rate=nr, cardinality=100,
                                   mean=50.0 if name=="amount" else None,
                                   std=10.0  if name=="amount" else None,
                                   p25=40.0  if name=="amount" else None,
                                   p75=60.0  if name=="amount" else None,
                                   min_val=5.0 if name=="amount" else None,
                                   max_val=95.0 if name=="amount" else None))
    return TableProfile(dataset="ecomm", table="orders",
                        row_count=row_count,
                        schema=[{"name":c.name,"type":c.dtype,"mode":"NULLABLE"} for c in cols],
                        columns=cols)


class TestNullRateDetection(unittest.TestCase):
    def setUp(self):
        mock_bq = MagicMock()
        mock_bq.get_daily_row_counts.side_effect = Exception("skip")
        self.detector = AnomalyDetector(connector=mock_bq)

    def test_critical_null_rate_flagged(self):
        profile = _make_profile(null_rates={"email": 0.25})
        report  = self.detector.detect(profile)
        critical = [a for a in report.anomalies if a.anomaly_type=="null_rate" and a.severity=="critical"]
        self.assertTrue(any("email" in a.column for a in critical))

    def test_medium_null_rate_flagged(self):
        profile = _make_profile(null_rates={"email": 0.10})
        report  = self.detector.detect(profile)
        medium  = [a for a in report.anomalies if a.anomaly_type=="null_rate" and a.severity=="medium"]
        self.assertTrue(any("email" in a.column for a in medium))

    def test_clean_table_no_null_anomalies(self):
        profile = _make_profile()
        report  = self.detector.detect(profile)
        null_anomalies = [a for a in report.anomalies if a.anomaly_type=="null_rate"]
        self.assertEqual(len(null_anomalies), 0)


class TestSchemaChange(unittest.TestCase):
    def setUp(self):
        mock_bq = MagicMock()
        mock_bq.get_daily_row_counts.side_effect = Exception("skip")
        self.detector = AnomalyDetector(connector=mock_bq)

    def _schema(self, cols):
        return [{"name": c, "type": "STRING", "mode": "NULLABLE"} for c in cols]

    def test_added_column_detected(self):
        curr = _make_profile(); curr.schema = self._schema(["a","b","c"])
        prev = _make_profile(); prev.schema = self._schema(["a","b"])
        report = self.detector.detect(curr, baseline=prev)
        schema_changes = [a for a in report.anomalies if a.anomaly_type=="schema_change"]
        self.assertTrue(any(a.column=="c" for a in schema_changes))

    def test_removed_column_flagged_critical(self):
        curr = _make_profile(); curr.schema = self._schema(["a","b"])
        prev = _make_profile(); prev.schema = self._schema(["a","b","c"])
        report = self.detector.detect(curr, baseline=prev)
        removed = [a for a in report.anomalies if a.anomaly_type=="schema_change" and a.severity=="critical"]
        self.assertTrue(any(a.column=="c" for a in removed))

    def test_no_changes_no_anomalies(self):
        curr = _make_profile(); curr.schema = self._schema(["a","b"])
        prev = _make_profile(); prev.schema = self._schema(["a","b"])
        report = self.detector.detect(curr, baseline=prev)
        schema_changes = [a for a in report.anomalies if a.anomaly_type=="schema_change"]
        self.assertEqual(len(schema_changes), 0)


class TestAnomalyReport(unittest.TestCase):
    def test_has_anomalies_false_when_empty(self):
        from src.anomaly_detector import AnomalyReport
        r = AnomalyReport(dataset="d", table="t", run_id="r1")
        self.assertFalse(r.has_anomalies)

    def test_critical_count_correct(self):
        from src.anomaly_detector import AnomalyReport
        r = AnomalyReport(dataset="d", table="t", run_id="r1",
            anomalies=[Anomaly("null_rate","col","desc","critical"),
                       Anomaly("null_rate","col","desc","medium")])
        self.assertEqual(r.critical_count, 1)


class TestZScoreAndDrift(unittest.TestCase):
    def setUp(self):
        mock_bq = MagicMock()
        mock_bq.get_daily_row_counts.side_effect = Exception("skip")
        self.detector = AnomalyDetector(connector=mock_bq)

    def test_extreme_value_flagged_by_zscore(self):
        profile = _make_profile()            # amount: mean 50, std 10, min 5 / max 95 -> |z| = 4.5
        report  = self.detector.detect(profile)
        z = [a for a in report.anomalies if a.anomaly_type == "zscore"]
        self.assertTrue(any(a.column == "amount" and abs(a.metric_value) == 4.5 for a in z))

    def test_values_within_threshold_not_flagged(self):
        profile = _make_profile()
        amount = next(c for c in profile.columns if c.name == "amount")
        amount.min_val, amount.max_val = 25.0, 75.0   # |z| = 2.5
        report = self.detector.detect(profile)
        self.assertFalse(any(a.anomaly_type == "zscore" for a in report.anomalies))

    def test_mean_drift_vs_baseline_flagged(self):
        prev = _make_profile()
        curr = _make_profile()
        next(c for c in curr.columns if c.name == "amount").mean = 85.0   # +3.5 baseline std
        report = self.detector.detect(curr, baseline=prev)
        drift = [a for a in report.anomalies if a.anomaly_type == "mean_drift"]
        self.assertTrue(any(a.column == "amount" and a.severity == "high" for a in drift))

    def test_stable_mean_no_drift(self):
        report = self.detector.detect(_make_profile(), baseline=_make_profile())
        self.assertFalse(any(a.anomaly_type == "mean_drift" for a in report.anomalies))



class TestVolumeDrift(unittest.TestCase):
    def setUp(self):
        self.detector = AnomalyDetector(connector=MagicMock())

    def _counts(self, values):
        import pandas as pd
        return pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=len(values)).date,
                             "row_count": values})

    def test_drop_flagged_critical(self):
        out = self.detector.volume_drift_from_counts(self._counts([1000, 1100, 900, 1000, 400]), 7)
        self.assertEqual([a.severity for a in out], ["critical"])

    def test_spike_flagged_high(self):
        out = self.detector.volume_drift_from_counts(self._counts([1000, 1000, 1000, 2500]), 7)
        self.assertEqual([a.severity for a in out], ["high"])

    def test_normal_day_not_flagged(self):
        self.assertEqual(self.detector.volume_drift_from_counts(self._counts([1000, 980, 1020, 1010]), 7), [])

    def test_not_enough_history(self):
        self.assertEqual(self.detector.volume_drift_from_counts(self._counts([1000, 10]), 7), [])


if __name__ == "__main__": unittest.main()
