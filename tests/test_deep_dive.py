import sys
import unittest
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ctr_posthoc import module
from deep_dive_common import top_budget,evidence_band,loss,TABULAR
d2=module('deep_dive_test','33_deep_dive_saved_predictions.py')


class DeepDiveTests(unittest.TestCase):
    def test_budget_ties_and_semantic_evidence(self):
        p=np.array([.5,.5,.5,.1]);ids=np.array([4,2,1,3])
        np.testing.assert_array_equal(top_budget(p,ids,.5),[False,True,True,False])
        np.testing.assert_array_equal(evidence_band(np.array([0,1,11,12,99,100])),['0','1-11','1-11','12-99','12-99','100+'])

    def test_common_miss_denominator_and_oracle_are_explicit(self):
        y=np.array([0,1,0,1,0,0,1,0,0,0],dtype=np.uint8)
        ctx=pl.DataFrame({'impression_id':np.arange(10),'clk':y})
        ps={name:np.linspace(.01,.1,10)+i*.001 for i,name in enumerate(list(TABULAR)+['attention_matched'])}
        errors=[];overlaps=[];corr=[];oracle=[]
        d2.common_errors(ctx,ps,'val',42,'attention_matched',errors,overlaps,corr,oracle,[])
        row=next(r for r in errors if r['policy']=='p>=0.5' and r['models_catching']==0)
        self.assertEqual(row['click_share'],1.)
        self.assertEqual(row['all_clicks'],3)
        for policy in set(r['policy'] for r in errors):
            part=[r for r in errors if r['policy']==policy]
            self.assertEqual(sum(r['rows'] for r in part),10)
            self.assertEqual(sum(r['clicks'] for r in part),3)
        bound=next(r['value'] for r in oracle if r['oracle']=='label_informed_min_logloss')
        self.assertLessEqual(bound,min(loss(y,p).mean() for p in ps.values()))


if __name__=='__main__':unittest.main()
