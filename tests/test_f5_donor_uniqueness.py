"""Regression fixtures for the frozen non-bijective assignment and its audit."""
import importlib.util
from pathlib import Path
import sys
import unittest
import polars as pl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from f5_sequences import donor_permutation
spec=importlib.util.spec_from_file_location('donor_audit',ROOT/'src/57_audit_f5_donor_uniqueness.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class DonorAuditTests(unittest.TestCase):
    def history(self):
        h=pl.DataFrame({'user_raw':[100,200,300,400,500],'length':[3]*5,
            'end':[3,6,9,12,15],'activity_bucket':[0]*5,'recency_bucket':[0]*5})
        return donor_permutation(h,20260926)

    def test_five_users_reproduce_duplicate(self):
        h=self.history()
        self.assertEqual(h['donor_user_raw'].n_unique(),4)
        self.assertEqual(h.filter(pl.col('donor_user_raw')==pl.col('user_raw')).height,0)

    def test_detect_reuse_and_exposure_denominators(self):
        h=self.history()
        refs=pl.DataFrame({'user_raw':[400,400,500,100,999], 'length':[3,3,3,3,0]})
        s,d,_=audit.assignment_statistics(h,refs)
        self.assertEqual((s['reuse_excess'],s['reused_donors'],s['max_indegree']),(1,1,2))
        self.assertEqual(s['targets_with_reused_donor'],2)
        self.assertEqual(s['affected_impressions'],3)
        self.assertAlmostEqual(s['affected_impression_share'],.6)
        self.assertEqual(s['unused_eligible_donors'],1)
        self.assertEqual(sum(row['donors'] for row in d),5)
        self.assertEqual(sum(row['donors']*row['indegree'] for row in d),5)

    def test_bijection_has_no_reuse(self):
        h=self.history().with_columns(pl.Series('donor_user_raw',[200,300,400,500,100]))
        s,_,_=audit.assignment_statistics(h,h.select('user_raw','length'))
        self.assertEqual(s['reuse_excess'],0)
        self.assertEqual(s['affected_impressions'],0)

if __name__=='__main__':unittest.main()
