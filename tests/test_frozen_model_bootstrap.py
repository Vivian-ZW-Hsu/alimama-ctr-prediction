"""Synthetic-fixture tests for the stage E3 frozen tabular bootstrap.

No saved prediction, model or frozen result is read. The kernel integration test
compiles src/cluster_bootstrap_metrics.cpp against tiny hand-built inputs.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ctr_posthoc import module, write_common, write_rank_model

bootstrap = module('frozen_model_bootstrap_test', '43_bootstrap_frozen_models.py')

PRIMARY = {'id': 1, 'left': 'a', 'right': 'b', 'metrics': ['auc', 'logloss'],
           'interpretation': 'confirmatory_primary'}
SECONDARY = {'id': 2, 'left': 'a', 'right': 'c', 'metrics': ['auc'],
             'interpretation': 'exploratory_secondary'}
INDEX = {'a': 0, 'b': 1, 'c': 2}


def replicates(values):
    """Build a (B+1, models, 4) array from per-model metric rows."""
    return np.asarray(values, dtype=float)


class PairedIntervalTests(unittest.TestCase):
    def test_delta_is_left_minus_right_without_orientation_flipping(self):
        # Model a has higher AUC and lower LogLoss than model b in every row.
        rng = np.random.default_rng(0)
        b_replicates = 200
        full = np.zeros((b_replicates + 1, 3, 4))
        full[:, 0, 0] = 0.65 + rng.normal(0, 1e-4, b_replicates + 1)   # a auc
        full[:, 1, 0] = 0.64 + rng.normal(0, 1e-4, b_replicates + 1)   # b auc
        full[:, 2, 0] = 0.60
        full[:, 0, 1] = 0.19 + rng.normal(0, 1e-5, b_replicates + 1)   # a logloss
        full[:, 1, 1] = 0.20 + rng.normal(0, 1e-5, b_replicates + 1)   # b logloss
        full[:, 2, 1] = 0.21

        rows, draws = bootstrap.paired_intervals(full, INDEX, [PRIMARY, SECONDARY], b_replicates)
        by_metric = {(r['id'], r['metric']): r for r in rows}

        auc = by_metric[(1, 'auc')]
        logloss = by_metric[(1, 'logloss')]
        # AUC: higher is better, so a better model gives a positive effect.
        self.assertGreater(auc['effect'], 0)
        # LogLoss: lower is better, and the sign is NOT flipped to "look" positive.
        self.assertLess(logloss['effect'], 0)
        self.assertAlmostEqual(auc['effect'], full[0, 0, 0] - full[0, 1, 0], places=12)
        self.assertAlmostEqual(logloss['effect'], full[0, 0, 1] - full[0, 1, 1], places=12)
        # Only the declared metrics of each comparison are produced.
        self.assertEqual(
            sorted(k for k in by_metric if k[0] == 2), [(2, 'auc')]
        )
        # Draw columns carry exactly B rows, excluding the unweighted replicate.
        self.assertEqual(len(draws['replicate']), b_replicates)
        self.assertEqual(len(draws['comparison_1_auc']), b_replicates)

    def test_pairing_uses_the_same_replicate_for_both_models(self):
        # If both models are identical in every replicate the paired delta must be
        # exactly zero everywhere, not merely zero on average. This fails if the
        # implementation ever compares across different replicate indices.
        rng = np.random.default_rng(7)
        b_replicates = 50
        shared = rng.normal(0.65, 0.01, b_replicates + 1)
        full = np.zeros((b_replicates + 1, 3, 4))
        full[:, 0, 0] = shared
        full[:, 1, 0] = shared
        rows, draws = bootstrap.paired_intervals(full, INDEX, [PRIMARY], b_replicates)
        auc = [r for r in rows if r['metric'] == 'auc'][0]
        self.assertEqual(auc['effect'], 0.0)
        np.testing.assert_array_equal(draws['comparison_1_auc'], np.zeros(b_replicates))
        self.assertEqual(auc['ci_low'], 0.0)
        self.assertEqual(auc['ci_high'], 0.0)
        self.assertFalse(auc['excludes_zero'])

    def test_interval_comes_from_resamples_and_p_has_declared_resolution(self):
        b_replicates = 99
        full = np.zeros((b_replicates + 1, 3, 4))
        # Row 0 is the point estimate; resamples are deliberately offset from it.
        full[0, 0, 0] = 0.50
        full[1:, 0, 0] = np.linspace(1.0, 2.0, b_replicates)
        rows, _ = bootstrap.paired_intervals(full, INDEX, [PRIMARY], b_replicates)
        auc = [r for r in rows if r['metric'] == 'auc'][0]
        self.assertEqual(auc['effect'], 0.50)
        # The interval must reflect rows 1..B, not row 0.
        self.assertGreater(auc['ci_low'], 0.9)
        self.assertLess(auc['ci_high'], 2.1)
        # Centered p-value resolution is 1/(B+1).
        self.assertGreaterEqual(auc['p_centered_two_sided'], 1 / (b_replicates + 1))

    def test_holm_corrects_only_confirmatory_primary_rows(self):
        rows = [
            {'interpretation': 'confirmatory_primary', 'p_centered_two_sided': 0.01,
             'holm_primary': None},
            {'interpretation': 'confirmatory_primary', 'p_centered_two_sided': 0.03,
             'holm_primary': None},
            {'interpretation': 'exploratory_secondary', 'p_centered_two_sided': 0.001,
             'holm_primary': None},
        ]
        corrected = bootstrap.apply_holm(rows)
        # Two primaries: Holm multiplies the smallest by 2 and enforces monotonicity.
        self.assertAlmostEqual(corrected[0]['holm_primary'], 0.02)
        self.assertAlmostEqual(corrected[1]['holm_primary'], 0.03)
        # The exploratory row keeps a null correction even though its p is smallest.
        self.assertIsNone(corrected[2]['holm_primary'])

    def test_non_finite_replicate_is_rejected_not_dropped(self):
        array = np.zeros((4, 2, 4))
        array[2, 1, 0] = np.nan
        with self.assertRaises(ValueError) as caught:
            bootstrap.validate_replicates(array, 'full bootstrap')
        self.assertIn('not silently discarded', str(caught.exception))
        # A fully finite array passes.
        bootstrap.validate_replicates(np.zeros((4, 2, 4)), 'full bootstrap')


class KernelSingleClassTests(unittest.TestCase):
    def test_single_class_draw_gives_undefined_auc_but_finite_logloss(self):
        # Justifies validate_replicates: the kernel reports NaN rather than a
        # fabricated AUC when a resample contains only one label.
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            binary = d / 'bootstrap'
            subprocess.run(
                ['clang++', '-O3', '-std=c++17', '-pthread',
                 str(ROOT / 'src/cluster_bootstrap_metrics.cpp'), '-o', str(binary)],
                check=True,
            )
            user = np.array([0, 0, 1, 1], dtype=np.uint32)
            labels = np.array([1, 1, 0, 1], dtype=np.uint8)   # user 0 is all positive
            predictions = np.array([0.8, 0.6, 0.3, 0.7])
            counts = np.bincount(user)
            # Only user 1 has both labels, so only it is GAUC-eligible.
            eligible_counts = np.array([0, counts[1]], dtype=np.uint32)
            write_common(d / 'common.bin', 4, 2, 1, counts, eligible_counts)
            write_rank_model(d / 'model_0.bin', user, labels, predictions, 2,
                             np.array([0.0, counts[1] * 1.0]))
            # Draw user 0 twice and user 1 never: the sample is single-class.
            np.array([2, 0], dtype=np.uint32).tofile(d / 'weights.bin')
            subprocess.run([str(binary), str(d), str(d / 'out.bin'), '0', '1', '1',
                            str(d / 'weights.bin')], check=True)
            result = np.fromfile(d / 'out.bin', dtype=np.float64)

        auc, logloss = result[0], result[1]
        self.assertTrue(np.isnan(auc), 'single-class AUC must be undefined, not 0 or 0.5')
        self.assertTrue(np.isfinite(logloss))
        expected = log_loss(
            np.repeat(labels[user == 0], 1), np.repeat(predictions[user == 0], 1),
            labels=[0, 1],
        )
        np.testing.assert_allclose(logloss, expected, rtol=1e-12)
        # The guard in validate_replicates would reject such a replicate.
        with self.assertRaises(ValueError):
            bootstrap.validate_replicates(result.reshape(1, 1, 4), 'draw')


if __name__ == '__main__':
    unittest.main()
