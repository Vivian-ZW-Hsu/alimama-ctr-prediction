import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sequence_ctr import SequenceResidual, gather_tokens


class SequenceTests(unittest.TestCase):
    def test_masks_gradients_neutrality_and_permutation(self):
        for pooling in ('mean','attention'):
            torch.manual_seed(42)
            model=SequenceResidual(torch.randn(10,8),pooling,0.0)
            candidate=torch.tensor([2,1,3,4])
            tokens=torch.tensor([[2,3,0,0],[1,0,1,0],[0,0,0,0],[4,5,6,0]])
            self.assertTrue(torch.equal(model(candidate,tokens),torch.zeros(4)))
            # Nonzero trained-like head must still not introduce an empty fallback.
            torch.nn.init.normal_(model.head[-1].weight)
            torch.nn.init.constant_(model.head[-1].bias,1.3)
            model.eval()
            output=model(candidate,tokens)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(output[1:3],torch.zeros(2)))
            torch.testing.assert_close(output,model(candidate,torch.cat([tokens,torch.zeros(4,3,dtype=torch.long)],dim=1)))
            torch.testing.assert_close(output,model(candidate,tokens[:,[3,1,0,2]]))
            _, pooled,_=model.representation(candidate,tokens)
            self.assertTrue(torch.equal(pooled[1:3],torch.zeros(2,8)))
            output.sum().backward()
            self.assertEqual(torch.count_nonzero(model.embedding.weight.grad[:2]).item(),0)
            self.assertGreater(torch.count_nonzero(model.embedding.weight.grad[2:]).item(),0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
            optimizer.step()
            model.assert_neutral()

    def test_gather_preserves_order_and_ignores_empty_pointer(self):
        categories=np.array([5,2,7,1,4],dtype=np.uint32)
        got=gather_tokens(categories,np.array([1,0,3]),np.array([2,0,2]),4)
        np.testing.assert_array_equal(got,[[2,7,0,0],[0,0,0,0],[1,4,0,0]])


if __name__=='__main__':
    unittest.main()
