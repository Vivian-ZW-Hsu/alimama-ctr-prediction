"""Synthetic-fixture tests for the stage E1 behavior namespace audit.

No raw table, Parquet copy, model or frozen result is read here.
"""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ctr_posthoc import module

audit = module('behavior_namespace_audit_test', '41_audit_behavior_namespace.py')


class NamespaceAuditTests(unittest.TestCase):
    def test_coverage_directions_are_not_symmetric_and_samples_are_unmatched(self):
        # Left has 4 ids, 3 of them shared; right has 6 ids, 3 of them shared.
        record = audit.coverage_record(
            'left.id', {1, 2, 3, 99},
            'right.id', {1, 2, 3, 40, 50, 60},
            sample_size=2,
        )
        self.assertEqual(record['left_distinct'], 4)
        self.assertEqual(record['right_distinct'], 6)
        self.assertEqual(record['shared_distinct'], 3)
        self.assertAlmostEqual(record['forward_coverage'], 3 / 4)
        self.assertAlmostEqual(record['reverse_coverage'], 3 / 6)
        # Forward and reverse must not be collapsed into one number.
        self.assertNotAlmostEqual(record['forward_coverage'], record['reverse_coverage'])
        # Samples contain only genuinely unmatched ids, truncated to sample_size.
        self.assertEqual(record['left_only_sample'], [99])
        self.assertEqual(record['right_only_sample'], [40, 50])
        self.assertEqual(record['right_only_count'], 3)
        for value in record['left_only_sample'] + record['right_only_sample']:
            self.assertNotIn(value, {1, 2, 3})

    def test_beijing_date_uses_utc_plus_eight_not_utc(self):
        # 1494000000 is 2017-05-05 16:00 UTC and 2017-05-06 00:00 Beijing;
        # 1494691186 is 2017-05-13 15:59:46 UTC and 2017-05-13 23:59:46 Beijing.
        frame = pl.LazyFrame({'time_stamp': [1494000000, 1494691186, 1494000000 - 1]})
        got = frame.select(audit.beijing_date().alias('d')).collect()['d'].to_list()
        self.assertEqual(got[0], date(2017, 5, 6))
        self.assertEqual(got[1], date(2017, 5, 13))
        # One second earlier still belongs to the previous Beijing day.
        self.assertEqual(got[2], date(2017, 5, 5))

    def test_ad_feature_null_token_parses_as_missing_integer(self):
        # ad_feature.csv writes a missing brand as the literal token NULL, which
        # fails Int64 parsing unless it is declared as a null value.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ad_feature.csv'
            path.write_text(
                'adgroup_id,cate_id,campaign_id,customer,brand,price\n'
                '1,10,100,5,777,9.0\n'
                '2,11,101,6,NULL,99.0\n'
            )
            with self.assertRaises(pl.exceptions.ComputeError):
                pl.scan_csv(path, schema=audit.AD_SCHEMA).collect()
            frame = pl.scan_csv(
                path, schema=audit.AD_SCHEMA, null_values=audit.NULL_VALUES
            ).collect()
        self.assertEqual(frame['brand'].to_list(), [777, None])
        self.assertEqual(frame['brand'].null_count(), 1)
        self.assertEqual(frame['brand'].dtype, pl.Int64)

    def test_verdict_applies_predeclared_thresholds_per_field(self):
        thresholds = audit.THRESHOLDS
        self.assertGreater(thresholds['ad_category_forward_coverage_min'], 0.5)

        def namespaces(category, user, brand):
            return {
                'category': {'forward_coverage': category},
                'user': {'forward_coverage': user},
                'brand': {'forward_coverage': brand},
            }

        passing = audit.verdict_for(namespaces(0.999, 0.995, 0.993))
        self.assertTrue(passing['f5_admissible'])
        self.assertTrue(passing['brand_token_admissible'])

        # Category below threshold blocks F5 outright.
        blocked = audit.verdict_for(namespaces(0.10, 0.995, 0.993))
        self.assertFalse(blocked['category_precondition_met'])
        self.assertFalse(blocked['f5_admissible'])

        # A failing user namespace also blocks F5.
        blocked_user = audit.verdict_for(namespaces(0.999, 0.10, 0.993))
        self.assertFalse(blocked_user['user_precondition_met'])
        self.assertFalse(blocked_user['f5_admissible'])

        # Brand is secondary: it downgrades tokens but does not block F5.
        category_only = audit.verdict_for(namespaces(0.999, 0.995, 0.10))
        self.assertTrue(category_only['f5_admissible'])
        self.assertFalse(category_only['brand_token_admissible'])
        self.assertIn('category-only', category_only['brand_policy'])

    def test_formal_window_excludes_sporadic_and_out_of_range_dates(self):
        self.assertEqual(audit.FORMAL_WINDOW_START, date(2017, 4, 22))
        self.assertEqual(audit.FORMAL_WINDOW_END_EXCLUSIVE, date(2017, 5, 14))
        frame = pl.LazyFrame(
            {
                'time_stamp': [
                    -2142988435,   # 1902, corrupt
                    1490572800,    # 2017-03-27 Beijing, sporadic
                    1492819200,    # 2017-04-22 Beijing, first formal day
                    1494691186,    # 2017-05-13 Beijing, last formal day
                    2099392509,    # 2036, corrupt
                ]
            }
        )
        kept = (
            frame.filter(
                (audit.beijing_date() >= audit.FORMAL_WINDOW_START)
                & (audit.beijing_date() < audit.FORMAL_WINDOW_END_EXCLUSIVE)
            )
            .select(audit.beijing_date().alias('d'))
            .collect()['d']
            .to_list()
        )
        self.assertEqual(kept, [date(2017, 4, 22), date(2017, 5, 13)])


if __name__ == '__main__':
    unittest.main()
