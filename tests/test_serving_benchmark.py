"""Synthetic-fixture tests for the serving benchmark scaffolding.

No model, prediction or frozen result is loaded. These cover the parts that can
silently produce a misleading table: the percentile helper, the adapter registry,
and the declared measurement plan.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ctr_posthoc import module

bench = module('serving_benchmark_test', '49_serving_benchmark.py')


class PercentileTests(unittest.TestCase):
    def test_percentile_matches_numpy_linear_interpolation(self):
        values = [float(v) for v in range(1, 101)]
        for fraction in (0.5, 0.95, 0.99):
            self.assertAlmostEqual(
                bench.percentile(values, fraction),
                float(np.quantile(np.asarray(values), fraction, method='linear')),
                places=12,
            )

    def test_tail_percentiles_are_ordered_and_sensitive_to_outliers(self):
        values = [1.0] * 99 + [1000.0]
        p50 = bench.percentile(values, 0.50)
        p95 = bench.percentile(values, 0.95)
        p99 = bench.percentile(values, 0.99)
        self.assertLessEqual(p50, p95)
        self.assertLessEqual(p95, p99)
        # A single slow call must move p99 but not p50, which is the whole point
        # of reporting a tail latency rather than a mean.
        self.assertEqual(p50, 1.0)
        self.assertGreater(p99, 1.0)


class RegistryTests(unittest.TestCase):
    def test_every_declared_model_has_a_distinct_key_and_family(self):
        keys = sorted(bench.ADAPTERS)
        self.assertEqual(
            keys,
            ['deepfm_f123', 'f4_mean_matched', 'f5_mean_matched', 'lightgbm_f123',
             'lr_f123', 'wide_deep_f123'],
        )
        families = [cls.family for cls in bench.ADAPTERS.values()]
        self.assertEqual(len(families), len(set(families)))
        for key, cls in bench.ADAPTERS.items():
            self.assertEqual(cls.key, key)
            # Each adapter must expose the three-method contract the runner calls.
            for method in ('load', 'prepare', 'infer'):
                self.assertTrue(callable(getattr(cls, method, None)), f'{key}.{method}')

    def test_sequence_adapters_declare_their_retained_length_and_share_the_backbone(self):
        f4 = bench.ADAPTERS['f4_mean_matched']
        f5 = bench.ADAPTERS['f5_mean_matched']
        self.assertEqual(f4.max_length, 20)
        self.assertEqual(f5.max_length, 200)
        # Both residuals run on the same frozen DeepFM checkpoint, so their cost
        # includes that backbone and their file size is the backbone's.
        self.assertEqual(f4.path, f5.path)
        self.assertEqual(f4.path, bench.ADAPTERS['deepfm_f123'].path)
        # F5 carries a second token field; F4 carries only the category.
        self.assertEqual(f4.event_columns, ['cate_idx'])
        self.assertEqual(f5.event_columns, ['cate_idx', 'type_idx'])

    def test_measurement_plan_is_fixed_and_reports_both_regimes(self):
        self.assertGreaterEqual(bench.SINGLE_WARMUP, 1)
        self.assertGreaterEqual(bench.SINGLE_REPEATS, 100)
        self.assertGreaterEqual(bench.BATCH_WARMUP, 1)
        self.assertEqual(bench.BATCH_SIZES, (1024, 8192))
        # Timing on a single common device keeps the families comparable.
        self.assertEqual(bench.SPLIT, 'val')


if __name__ == '__main__':
    unittest.main()
