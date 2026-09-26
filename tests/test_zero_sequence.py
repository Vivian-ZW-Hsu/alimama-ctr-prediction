import io
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sequence_ctr import SequenceResidual
from zero_sequence_ctr import ZeroSequenceResidual


class ZeroSequenceTests(unittest.TestCase):
    def test_bias_masks_updates_and_checkpoint_cannot_create_signal(self):
        model=ZeroSequenceResidual(torch.randn(9,8),'mean')
        original=SequenceResidual(torch.randn(9,8),'mean')
        self.assertEqual(model.head[0].in_features,original.head[0].in_features)
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in original.parameters()))
        # Deliberately nonzero weights and biases: output still must be zero.
        with torch.no_grad():
            for p in model.head.parameters():p.fill_(0.4)
        q=torch.tensor([0,1,2,3,4])
        h=torch.tensor([[0,0,0],[1,1,0],[2,3,4],[0,0,0],[7,1,8]])
        optim=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
        for _ in range(3):
            model.train();optim.zero_grad()
            query,pooled,mask=model.representation(q,h)
            self.assertEqual(torch.count_nonzero(query).item()+torch.count_nonzero(pooled).item()+torch.count_nonzero(mask).item(),0)
            delta=model(q,h)
            self.assertTrue(torch.equal(delta,torch.zeros(5)))
            (delta+1).square().sum().backward()
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad).item()==0 for p in model.parameters()))
            optim.step();model.assert_neutral()
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        reloaded=ZeroSequenceResidual(torch.randn(9,8),'mean')
        reloaded.load_state_dict(torch.load(buffer,weights_only=True));reloaded.eval()
        self.assertTrue(torch.equal(reloaded(q,h),torch.zeros(5)))
        self.assertTrue(torch.equal(reloaded(q,torch.ones_like(h)),torch.zeros(5)))


if __name__=='__main__':unittest.main()
