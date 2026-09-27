"""Synthetic-fixture tests for the experiment registry classification.

Misclassifying a directory would quietly move an experiment into the wrong stage
of the published index, so the prefix rules are tested directly. Nothing on disk
is read.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ctr_posthoc import module

registry = module('experiment_registry_test', '50_build_experiment_registry.py')


class ClassificationTests(unittest.TestCase):
    def test_longest_matching_prefix_wins(self):
        # 'lr_f1' also prefixes 'lr_f1_f2_f3', so the most specific rule must win
        # or every LR information layer would collapse into one budget.
        self.assertEqual(registry.classify('lr_f1')[1], 'F1')
        self.assertEqual(registry.classify('lr_f1_f2')[1], 'F1+F2')
        self.assertEqual(registry.classify('lr_f1_f2_f3')[1], 'F1+F2+F3')
        for name in ('lr_f1', 'lr_f1_f2', 'lr_f1_f2_f3'):
            self.assertEqual(registry.classify(name)[0], 'tabular_information_ablation')

    def test_f4_and_f5_sequence_runs_are_separated(self):
        f4_stage, f4_budget, _ = registry.classify('seq_mean_matched_l20_lr500_s42_v1')
        f5_stage, f5_budget, _ = registry.classify('f5b_mean_matched_l200_lr500_s42_v1')
        self.assertEqual(f4_stage, 'f4_sequence_validation')
        self.assertEqual(f5_stage, 'f5_sequence_validation')
        self.assertEqual(f4_budget, 'F1+F2+F3+F4')
        self.assertEqual(f5_budget, 'F1+F2+F3+F5')
        # The once-only F4 test stage is distinct from its validation stage.
        self.assertEqual(registry.classify('sequence_test_v2')[0], 'f4_sequence_test')

    def test_tuning_and_data_stages_are_not_confused_with_formal_runs(self):
        self.assertEqual(registry.classify('lgbm_f123_leaves31')[0], 'tuning')
        self.assertEqual(registry.classify('deepfm_f123_compact')[0], 'tuning')
        # The frozen single-seed DeepFM is a benchmark run, not a tuning run.
        self.assertEqual(registry.classify('deepfm_f1_f2_f3')[0], 'tabular_benchmark')
        self.assertEqual(registry.classify('f5_behavior_v1')[0], 'data')
        self.assertEqual(registry.classify('f5_context_v1')[0], 'context_cache')
        self.assertEqual(registry.classify('behavior_namespace_audit_v1')[0], 'data')
        self.assertEqual(registry.classify('frozen_model_bootstrap_v1')[0], 'frozen_model_bootstrap')

    def test_unknown_directories_are_reported_not_dropped(self):
        stage, budget, note = registry.classify('something_nobody_declared')
        self.assertEqual(stage, 'unclassified')
        self.assertEqual(budget, '-')
        self.assertIn('no declared prefix rule', note)

    def test_every_declared_rule_classifies_its_own_prefix(self):
        for stage, prefix, budget, _ in registry.STAGES:
            observed_stage, observed_budget, _ = registry.classify(prefix)
            # A rule that cannot classify its own prefix is shadowed by a longer one.
            self.assertNotEqual(observed_stage, 'unclassified', prefix)
            self.assertEqual((observed_stage, observed_budget), (stage, budget), prefix)


if __name__ == '__main__':
    unittest.main()
