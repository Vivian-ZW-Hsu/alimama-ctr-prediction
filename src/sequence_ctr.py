"""Category history residual with a strictly neutral empty/unknown representation."""
import numpy as np
import torch
from torch import nn


def gather_tokens(event_categories, starts, lengths, max_length=20):
    mask = np.arange(max_length)[None, :] < lengths[:, None]
    indices = starts[:, None] + np.arange(max_length)[None, :]
    tokens = event_categories[np.where(mask, indices, 0)].astype(np.int64)
    tokens[~mask] = 0
    return tokens


class SequenceResidual(nn.Module):
    def __init__(self, initial_embedding, pooling='mean', dropout=0.3):
        super().__init__()
        if pooling not in ('mean', 'attention'):
            raise ValueError(pooling)
        self.pooling = pooling
        self.embedding = nn.Embedding.from_pretrained(initial_embedding.clone(), freeze=False, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight[:2].zero_()
        dim = initial_embedding.shape[1]
        # Create the shared head before attention so equal seeds initialize it identically.
        self.head = nn.Sequential(nn.Linear(dim*4,64),nn.ReLU(),nn.Dropout(dropout),
                                  nn.Linear(64,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.attention = (nn.Sequential(nn.Linear(dim*4,32),nn.ReLU(),nn.Linear(32,1))
                          if pooling == 'attention' else None)

    def representation(self, candidate, history):
        mask = history > 1
        q = self.embedding(candidate) * (candidate > 1).unsqueeze(-1)
        h = self.embedding(history) * mask.unsqueeze(-1)
        if self.pooling == 'mean':
            pooled = h.sum(dim=1) / mask.sum(dim=1,keepdim=True).clamp_min(1)
        else:
            expanded = q[:,None,:].expand_as(h)
            score = self.attention(torch.cat([expanded,h,expanded-h,expanded*h],dim=-1)).squeeze(-1)
            weights = torch.softmax(score.masked_fill(~mask, -1e4), dim=1) * mask
            weights = weights / weights.sum(dim=1,keepdim=True).clamp_min(1e-12)
            pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        return q, pooled, mask.any(dim=1)

    def forward(self, candidate, history):
        q, pooled, has_history = self.representation(candidate, history)
        value = self.head(torch.cat([q,pooled,q-pooled,q*pooled],dim=1)).squeeze(-1)
        return value * has_history

    def assert_neutral(self):
        assert torch.count_nonzero(self.embedding.weight[:2]).item() == 0
