import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

spec = importlib.util.spec_from_file_location('evaluator', Path(__file__).resolve().parents[1]/'src/10_evaluate_predictions.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluatorSplitTests(unittest.TestCase):
    def test_val_only_with_no_test_files_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('features','predictions/run','mappings','results'):
                (root/name).mkdir(parents=True)
            pl.DataFrame({'impression_id':[0,1,2,3], 'clk':[0,1,0,1],
                          'user_raw':[1,1,2,2], 'adgroup_id_raw':[10,11,10,11],
                          'pid_raw':[1]*4,'hour':[1]*4,'event_date_bj':['2017-05-12']*4
                          }).write_parquet(root/'features/val.parquet')
            pl.DataFrame({'impression_id':[0,1,2,3],'clk':[0,1,0,1],
                          'prediction':[0.1,0.8,0.2,0.7]}).write_parquet(root/'predictions/run/val_predictions.parquet')
            pl.DataFrame({'user_raw':[1]}).write_parquet(root/'mappings/user_raw_mapping.parquet')
            pl.DataFrame({'adgroup_id_raw':[10]}).write_parquet(root/'mappings/adgroup_id_raw_mapping.parquet')
            with patch.multiple(evaluator,FEATURE_DIR=root/'features',PREDICTION_ROOT=root/'predictions',
                                ENCODER_MAPPING_DIR=root/'mappings',RESULT_ROOT=root/'results'), \
                 patch.object(sys,'argv',['evaluator','--run-name','run','--splits','val']), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluator.main()
                with self.assertRaises(FileExistsError):evaluator.main()
            manifest=json.loads((root/'results/run/evaluation/evaluation_manifest.json').read_text())
            self.assertEqual(manifest['splits'],['val'])
            self.assertEqual(pl.read_csv(root/'results/run/evaluation/overall_metrics.csv')['auc'][0],1.0)

    def test_defaults_and_reject_calibrated_val(self):
        with patch.object(sys,'argv',['evaluator','--run-name','run']):
            self.assertIsNone(evaluator.parse_args().splits)
        with patch.object(sys,'argv',['evaluator','--run-name','run','--splits','val','--prediction-stage','calibrated']):
            with self.assertRaises(ValueError):evaluator.validate_args(evaluator.parse_args())


if __name__=='__main__':unittest.main()
