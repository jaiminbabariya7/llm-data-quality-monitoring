"""Unit tests for DataProfiler."""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))

from src.data_profiler import (DataProfiler, ColumnProfile, TableProfile,
                               save_profile, load_profile)


def _make_df(n=500, null_frac=0.0):
    np.random.seed(42)
    df = pd.DataFrame({
        "order_id":    range(n),
        "amount":      np.random.normal(50, 15, n),
        "customer_id": np.random.choice(["c1","c2","c3","c4"], n),
        "status":      np.random.choice(["paid","refund","pending"], n),
    })
    if null_frac > 0:
        mask = np.random.rand(n) < null_frac
        df.loc[mask, "customer_id"] = None
    return df

def _mock_schema():
    return [{"name": c, "type": "STRING" if c in ("customer_id","status","order_id") else "FLOAT64", "mode": "NULLABLE"}
            for c in ["order_id","amount","customer_id","status"]]


class TestColumnProfile(unittest.TestCase):
    def setUp(self):
        self.profiler = DataProfiler(connector=MagicMock())
        self.df = _make_df()

    def test_numeric_column_has_stats(self):
        col = self.profiler._profile_column(self.df, {"name":"amount","type":"FLOAT64","mode":"NULLABLE"})
        self.assertIsNotNone(col.mean)
        self.assertIsNotNone(col.std)
        self.assertIsNotNone(col.p25)
        self.assertIsNotNone(col.p75)

    def test_string_column_no_numeric_stats(self):
        col = self.profiler._profile_column(self.df, {"name":"status","type":"STRING","mode":"NULLABLE"})
        self.assertIsNone(col.mean)

    def test_null_rate_zero_for_clean_column(self):
        col = self.profiler._profile_column(self.df, {"name":"amount","type":"FLOAT64","mode":"NULLABLE"})
        self.assertEqual(col.null_rate, 0.0)

    def test_null_rate_detected_correctly(self):
        df = _make_df(null_frac=0.2)
        col = self.profiler._profile_column(df, {"name":"customer_id","type":"STRING","mode":"NULLABLE"})
        self.assertGreater(col.null_rate, 0.15)
        self.assertLess(col.null_rate, 0.25)

    def test_cardinality_correct(self):
        col = self.profiler._profile_column(self.df, {"name":"status","type":"STRING","mode":"NULLABLE"})
        self.assertEqual(col.cardinality, 3)

    def test_top_values_not_empty(self):
        col = self.profiler._profile_column(self.df, {"name":"status","type":"STRING","mode":"NULLABLE"})
        self.assertGreater(len(col.top_values), 0)


class TestNullFlagging(unittest.TestCase):
    def test_flag_null_issues_above_threshold(self):
        profiler = DataProfiler(connector=MagicMock())
        df = _make_df(null_frac=0.3)
        schema = _mock_schema()
        cols = [profiler._profile_column(df, s) for s in schema]
        profile = TableProfile(dataset="d", table="t", row_count=500,
                               schema=schema, columns=cols)
        issues = profiler.flag_null_issues(profile)
        self.assertTrue(any("customer_id" in i for i in issues))

    def test_no_issues_when_data_clean(self):
        profiler = DataProfiler(connector=MagicMock())
        df = _make_df()
        schema = _mock_schema()
        cols = [profiler._profile_column(df, s) for s in schema]
        profile = TableProfile(dataset="d", table="t", row_count=500,
                               schema=schema, columns=cols)
        self.assertEqual(profiler.flag_null_issues(profile), [])


class TestProfileStore(unittest.TestCase):
    def test_round_trip_and_first_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_profile("d", "t", store_dir=d))   # first run: no baseline
            col = ColumnProfile(name="amount", dtype="FLOAT64", null_rate=0.0,
                                cardinality=10, mean=50.0, std=10.0)
            prof = TableProfile(dataset="d", table="t", row_count=5,
                                schema=[{"name": "amount", "type": "FLOAT64"}],
                                columns=[col], profiled_at="2026-01-01T00:00:00Z")
            save_profile(prof, store_dir=d)
            loaded = load_profile("d", "t", store_dir=d)
            self.assertEqual(loaded.row_count, 5)
            self.assertEqual(loaded.columns[0].mean, 50.0)
            self.assertIsInstance(loaded.columns[0], ColumnProfile)


if __name__ == "__main__": unittest.main()
