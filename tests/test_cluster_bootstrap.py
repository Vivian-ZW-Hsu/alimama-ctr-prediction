import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ctr_posthoc import write_common,write_rank_model,holm,length_labels,category_overlap


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.directory=Path(cls.tmp.name)
        cls.binary=cls.directory/'bootstrap'
        subprocess.run(['clang++','-O3','-std=c++17','-pthread',str(ROOT/'src/cluster_bootstrap_metrics.cpp'),'-o',str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_exact_ties_whole_user_weights_and_thread_reproducibility(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);u=np.repeat(np.arange(3),3).astype(np.uint32)
            y=np.array([0,1,0,1,0,1,0,1,1],dtype=np.uint8)
            p=np.array([.1,.3,.3,.7,.7,.2,.2,.9,.9])
            counts=np.bincount(u);ga=np.array([roc_auc_score(y[u==i],p[u==i])*counts[i] for i in range(3)])
            write_common(d/'common.bin',9,3,1,counts,counts)
            write_rank_model(d/'model_0.bin',u,y,p,3,ga)
            weights=np.array([2,0,1],dtype=np.uint32);weights.tofile(d/'weights.bin')
            subprocess.run([str(self.binary),str(d),str(d/'weighted.bin'),'0','20260927','1',str(d/'weights.bin')],check=True)
            result=np.fromfile(d/'weighted.bin',dtype=np.float64)
            ix=np.repeat(np.arange(len(y)),weights[u])
            expected=[roc_auc_score(y[ix],p[ix]),log_loss(y[ix],p[ix]),average_precision_score(y[ix],p[ix]),np.dot(weights,ga)/np.dot(weights,counts)]
            np.testing.assert_allclose(result,expected,rtol=1e-13,atol=1e-14)
            for threads in (1,4):
                subprocess.run([str(self.binary),str(d),str(d/f'repeat{threads}.bin'),'20','20260927',str(threads)],check=True)
            self.assertEqual((d/'repeat1.bin').read_bytes(),(d/'repeat4.bin').read_bytes())

    def test_holm_and_group_boundaries_and_raw_category_unknowns(self):
        np.testing.assert_allclose(holm([.03,.01,.2]),[.06,.03,.2])
        np.testing.assert_array_equal(length_labels(np.array([0,1,2,5,6,10,11,19,20])),['0','1','2-5','2-5','6-10','6-10','11-19','11-19','20'])
        # Different raw unknown categories must not match just because encoded ID=1.
        actual=category_overlap(np.array([100,200,300]),np.array([0,1,0,2]),np.array([1,1,0,1]),np.array([999,200,100,-1]))
        np.testing.assert_array_equal(actual,[False,True,False,False])


if __name__=='__main__':unittest.main()
