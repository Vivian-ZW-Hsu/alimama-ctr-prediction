"""Small arithmetic fixtures; no saved experiment is executed."""
import importlib.util
from pathlib import Path
import statistics
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_audit',ROOT/'src/58_audit_e1_e6_release.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ReleaseArithmeticTests(unittest.TestCase):
    def test_holm_preserves_original_order(self):
        self.assertEqual(audit.holm([.04,.01,.03]),[.06,.03,.06])

    def test_seed_summary_rejects_population_std(self):
        raw=[{'condition':'fixture','seed':str(seed),'auc':str(value)} for seed,value in zip([42,2026,3407],[.5,.6,.7])]
        summary=[{'condition':'fixture','seeds':'3','auc_mean':'.6','auc_std':str(statistics.stdev([.5,.6,.7]))}]
        audit.check_summary(raw,summary)
        summary[0]['auc_std']=str(statistics.pstdev([.5,.6,.7]))
        with self.assertRaises(AssertionError):audit.check_summary(raw,summary)

    def test_seed_summary_rejects_duplicate_seed(self):
        raw=[{'condition':'fixture','seed':'42','auc':'.6'}]*3
        with self.assertRaises(AssertionError):
            audit.check_summary(raw,[{'condition':'fixture','seeds':'3','auc_mean':'.6','auc_std':'0'}])

if __name__=='__main__':unittest.main()
