"""Hard-zero residual with the same downstream dimensions and optimizer interface."""
import torch
from sequence_ctr import SequenceResidual


class ZeroSequenceResidual(SequenceResidual):
    def representation(self, candidate, history):
        zero = self.embedding.weight.new_zeros((candidate.shape[0], self.embedding.embedding_dim))
        return zero, zero, torch.zeros_like(candidate, dtype=torch.bool)

    def forward(self, candidate, history):
        q, pooled, _ = self.representation(candidate, history)
        # Execute the same 4*d -> 64 -> 32 -> 1 head and dropout, but erase
        # its entire output, including all biases. masked_fill also blocks
        # gradients, without relying on learned weights being zero.
        value = self.head(torch.cat([q, pooled, q-pooled, q*pooled], dim=1)).squeeze(-1)
        return value.masked_fill(torch.ones_like(value, dtype=torch.bool), 0.0)
