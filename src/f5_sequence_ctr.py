"""F5 residual: category history plus behavior type, on the frozen DeepFM context.

Differences from the F4 residual, all deliberate and minimal:

* Each history position carries two tokens, the event category and its behavior
  type. The type embedding is **added** to the category embedding rather than
  concatenated, so the pooled representation and the 4d -> 64 -> 32 -> 1 head keep
  exactly F4's dimensions. Capacity parity between mean and attention, and
  comparability with the F4 study, are preserved.
* The type embedding is **zero-initialised**, so at step zero an F5 model is
  numerically identical to a category-only model and any type contribution is
  learned rather than injected.
* The validity mask stays `category > 1`, the same rule F4 used. An event whose
  category falls outside the train ad vocabulary is treated as carrying no
  information even though its type is known. This keeps the only intended
  difference between F4 and F5 the history source itself; it discards 4.30% of
  in-window tokens, which is recorded rather than optimised away.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sequence_ctr import SequenceResidual


def gather_f5_tokens(categories, types, starts, lengths, max_length=200):
    """Expand one minibatch of pointers into category and type token matrices.

    Only `len(starts) x max_length` is ever allocated, never `N x max_length`.
    """
    mask = np.arange(max_length)[None, :] < lengths[:, None]
    indices = starts[:, None] + np.arange(max_length)[None, :]
    safe = np.where(mask, indices, 0)
    category_tokens = categories[safe].astype(np.int64)
    type_tokens = types[safe].astype(np.int64)
    category_tokens[~mask] = 0
    type_tokens[~mask] = 0
    return category_tokens, type_tokens


class F5SequenceResidual(SequenceResidual):
    def __init__(self, initial_embedding, pooling="mean", dropout=0.3,
                 type_vocabulary_size=6):
        super().__init__(initial_embedding, pooling=pooling, dropout=dropout)
        dim = initial_embedding.shape[1]
        self.type_embedding = nn.Embedding(type_vocabulary_size, dim, padding_idx=0)
        with torch.no_grad():
            self.type_embedding.weight.zero_()

    def representation(self, candidate, history, history_type):
        # PAD (0) and UNKNOWN (1) categories are both invalid, as in F4.
        mask = history > 1
        q = self.embedding(candidate) * (candidate > 1).unsqueeze(-1)
        tokens = (self.embedding(history) + self.type_embedding(history_type))
        h = tokens * mask.unsqueeze(-1)
        if self.pooling == "mean":
            pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            expanded = q[:, None, :].expand_as(h)
            score = self.attention(
                torch.cat([expanded, h, expanded - h, expanded * h], dim=-1)
            ).squeeze(-1)
            weights = torch.softmax(score.masked_fill(~mask, -1e4), dim=1) * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        return q, pooled, mask.any(dim=1)

    def forward(self, candidate, history, history_type):
        q, pooled, has_history = self.representation(candidate, history, history_type)
        value = self.head(torch.cat([q, pooled, q - pooled, q * pooled], dim=1)).squeeze(-1)
        return value * has_history

    def assert_neutral(self):
        super().assert_neutral()
        # PAD and UNKNOWN type rows must remain exactly zero.
        assert torch.count_nonzero(self.type_embedding.weight[:2]).item() == 0


class ZeroF5SequenceResidual(F5SequenceResidual):
    """Hard-zero control with the same dimensions and optimizer interface."""

    def representation(self, candidate, history, history_type):
        zero = self.embedding.weight.new_zeros(
            (candidate.shape[0], self.embedding.embedding_dim)
        )
        return zero, zero, torch.zeros_like(candidate, dtype=torch.bool)

    def forward(self, candidate, history, history_type):
        q, pooled, _ = self.representation(candidate, history, history_type)
        # Run the identical head and dropout, then erase the whole output including
        # every bias. masked_fill also blocks gradients, so the control cannot
        # create signal even if weights drift.
        value = self.head(torch.cat([q, pooled, q - pooled, q * pooled], dim=1)).squeeze(-1)
        return value.masked_fill(torch.ones_like(value, dtype=torch.bool), 0.0)
