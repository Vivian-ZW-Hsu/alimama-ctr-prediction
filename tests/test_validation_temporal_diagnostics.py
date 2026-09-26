import sys
import unittest
from pathlib import Path
import datetime as dt
import numpy as np
import polars as pl
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ctr_posthoc import module
d3=module('validation_component_test','34_validation_component_diagnostics.py')
d4=module('temporal_repeat_test','35_temporal_repeat_diagnostics.py')


class DiagnosticTests(unittest.TestCase):
    def test_components_reconstruct_and_dense_masks_respect_information_layers(self):
        torch.manual_seed(42)
        model=d3.tabular.TabularCTRModel('deepfm',[5,6],3,4,[8],0).eval()
        c=torch.tensor([[1,2],[2,3],[3,4]]);dense=torch.randn(3,3)
        with torch.no_grad():
            w,f,d=d3.component_logits(model,c,dense)
            torch.testing.assert_close(w+d+f,model(c,dense))
        self.assertEqual(d3.layer('user_past_ctr',{'user_past_ctr'}),'F3')
        self.assertEqual(d3.layer('log1p_pv_count_14d',set()),'F2')
        self.assertEqual(d3.layer('log_price',set()),'F1')

    def test_repeat_counts_exclude_whole_current_day(self):
        daily=pl.DataFrame({'user_raw':[1,1,1,2],'adgroup_id_raw':[7,7,7,7],
            'event_date_bj':[dt.date(2017,5,d) for d in [8,6,7,7]],'pair_rows':[5,2,3,4]})
        actual=d4.prior_pair_counts(daily)
        np.testing.assert_array_equal(actual.filter(pl.col('user_raw')==1)['prior_pair_rows'].to_numpy(),[0,2,5])
        self.assertEqual(actual.filter(pl.col('user_raw')==2)['prior_pair_rows'][0],0)


if __name__=='__main__':unittest.main()
