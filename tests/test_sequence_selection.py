import importlib.util
from pathlib import Path
import sys
import unittest

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location('selection',SRC/'23_select_sequence_configuration.py')
selection=importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def test_common_rate_uses_both_mechanisms_and_fixed_tie_break(self):
        rows=[{'learning_rate':rate,'pooling':pooling,'log_loss':loss}
              for rate,pooling,loss in [(5e-4,'mean',0.1),(5e-4,'attention',0.5),
                                        (2e-4,'mean',0.2),(2e-4,'attention',0.2)]]
        self.assertEqual(selection.select_common(rows)[0],2e-4)
        for row in rows:row['log_loss']=0.2
        self.assertEqual(selection.select_common(rows)[0],5e-4)
        with self.assertRaises(AssertionError):selection.select_common(rows[:-1])


if __name__=='__main__':unittest.main()
