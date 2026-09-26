import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ctr_posthoc import module, calibration, metrics

analysis = module('posthoc_calibration_test', '30_calibrate_diagnose_sequences.py')
summary = module('posthoc_summary_test', '31_summarize_prediction_analysis.py')


class CalibrationTests(unittest.TestCase):
    def test_selection_uses_only_hash_holdout_then_refits_full_validation(self):
        rng = np.random.default_rng(45)
        ids = np.arange(500, dtype=np.uint64)
        p = rng.uniform(.02, .8, size=500)
        labels = rng.binomial(1, p*.7).astype(np.uint8)
        holdout = calibration.deterministic_selection_mask(ids, 5)
        with patch.object(calibration, 'fit_platt', wraps=calibration.fit_platt) as fp, \
             patch.object(calibration, 'fit_isotonic', wraps=calibration.fit_isotonic) as fi:
            method, final, rows, nf, ns = analysis.select_and_refit(ids, labels, p)
            for fitter in (fp, fi):
                np.testing.assert_array_equal(fitter.call_args_list[0].args[0], labels[~holdout])
                np.testing.assert_array_equal(fitter.call_args_list[0].args[1], p[~holdout])
            chosen = fp if method == 'platt' else fi
            np.testing.assert_array_equal(chosen.call_args_list[-1].args[0], labels)
            self.assertEqual(nf+ns, len(ids))
            self.assertTrue(all(r['rows'] == holdout.sum() for r in rows))
            candidates = [r for r in rows if r['method'] != 'raw']
            self.assertEqual(method, min(candidates, key=lambda r: r['logloss'])['method'])
            transformed = calibration.apply_platt(final, p) if method == 'platt' else final.predict(p)
            self.assertTrue(np.isfinite(transformed).all())

    def test_single_zero_reference_and_pairing_not_seed_replication(self):
        rows = []
        for condition, seeds in [('mean_matched', [42, 2026, 3407]),
                                 ('mean_mismatched', [42, 2026, 3407]),
                                 ('attention_matched', [42, 2026, 3407]),
                                 (summary.ZERO, [42])]:
            for i, seed in enumerate(seeds):
                rows.append({'condition': condition, 'seed': seed, 'stage': 'raw', 'split': 'test',
                             'group': 'history_bin', 'value': '0', 'rows': 30, 'clicks': 3,
                             'actual_ctr': .1, **{m: .5+i*.01 for m in summary.METRICS}})
        frame = pl.DataFrame(rows)
        result = summary.summarize(frame, ['condition'], summary.METRICS)
        zero = result.filter(pl.col('condition') == summary.ZERO).row(0, named=True)
        self.assertEqual(zero['predictors'], 1)
        self.assertIsNone(zero['auc_std'])
        differences = summary.group_differences(frame)
        paired = differences.filter((pl.col('left') == 'mean_matched') & (pl.col('right') == summary.ZERO))
        np.testing.assert_allclose(paired['auc'].to_numpy(), [0, .01, .02], atol=1e-15)
        self.assertTrue((paired['rows'].to_numpy() == 30).all())

    def test_single_class_group_auc_is_undefined_but_logloss_valid(self):
        r = metrics(pl.DataFrame({'clk': [1, 1], 'prediction': [.1, .2]}))
        self.assertTrue(np.isnan(r['auc']))
        self.assertTrue(np.isfinite(r['logloss']))


if __name__ == '__main__':
    unittest.main()
