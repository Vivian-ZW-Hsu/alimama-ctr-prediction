import importlib.util
import sys
from pathlib import Path
import unittest

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location('report',SRC/'27_report_sequence_test.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class ReportingTests(unittest.TestCase):
    def test_paired_direction_sample_std_and_reference_exclusion(self):
        conditions=['zero','mean','mean_mismatch','attention','attention_mismatch']
        plan={'conditions':conditions,'seeds':[42,2026,3407],
              'comparisons':[{'id':1,'priority':'Primary','left':'mean','right':'zero'}]}
        rows=[]
        for c in conditions:
            for i,seed in enumerate(plan['seeds']):
                value=0.5+(i*0.01 if c=='mean' else 0)
                rows.append({'condition':c,'seed':seed,**{k:value for k in report.METRICS}})
        frame,summary,paired,comparisons=report.summarize(rows,plan)
        self.assertEqual(frame.height,15)
        self.assertAlmostEqual(comparisons['delta_auc_mean'][0],0.01)
        self.assertAlmostEqual(comparisons['delta_auc_std'][0],0.01)
        self.assertEqual(summary.filter(summary['condition']=='zero')['auc_std'][0],0.)
        with self.assertRaises(AssertionError):report.summarize(rows+[rows[0]],plan)


if __name__=='__main__':unittest.main()
