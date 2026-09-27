"""Synthetic-fixture tests for the F5 residual (category + behavior type).

No cache, event store, checkpoint or frozen result is read.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sequence_ctr import SequenceResidual
from f5_sequence_ctr import (
    F5SequenceResidual, ZeroF5SequenceResidual, gather_f5_tokens,
)

DIM = 4
VOCAB = 10


def embedding_fixture():
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, DIM)
    weight[:2].zero_()
    return weight


class GatherTests(unittest.TestCase):
    def test_gather_preserves_order_pads_and_ignores_empty_pointers(self):
        categories = np.array([9, 5, 6, 7, 8], dtype=np.uint16)
        types = np.array([2, 3, 4, 5, 2], dtype=np.uint8)
        starts = np.array([1, 0], dtype=np.uint32)
        lengths = np.array([3, 0], dtype=np.uint8)
        cat_tokens, type_tokens = gather_f5_tokens(
            categories, types, starts, lengths, max_length=4
        )
        # Row 0 keeps the stored order starting at index 1, then pads with zeros.
        np.testing.assert_array_equal(cat_tokens[0], [5, 6, 7, 0])
        np.testing.assert_array_equal(type_tokens[0], [3, 4, 5, 0])
        # An empty history is all PAD on both fields regardless of its pointer.
        np.testing.assert_array_equal(cat_tokens[1], [0, 0, 0, 0])
        np.testing.assert_array_equal(type_tokens[1], [0, 0, 0, 0])
        self.assertEqual(cat_tokens.dtype, np.int64)
        self.assertEqual(type_tokens.dtype, np.int64)


class F5ResidualTests(unittest.TestCase):
    def setUp(self):
        self.weight = embedding_fixture()
        self.candidate = torch.tensor([5, 6])
        self.history = torch.tensor([[7, 8, 0, 0], [9, 0, 0, 0]])
        self.types = torch.tensor([[2, 3, 0, 0], [5, 0, 0, 0]])

    def test_zero_initialised_type_embedding_matches_category_only_at_init(self):
        for pooling in ('mean', 'attention'):
            torch.manual_seed(1)
            f5 = F5SequenceResidual(self.weight, pooling=pooling).eval()
            torch.manual_seed(1)
            f4 = SequenceResidual(self.weight, pooling=pooling).eval()
            with torch.no_grad():
                a = f5(self.candidate, self.history, self.types)
                b = f4(self.candidate, self.history)
            # The type embedding starts at zero, so F5 begins numerically identical
            # to the category-only residual; any type effect must be learned.
            torch.testing.assert_close(a, b)

    def test_type_tokens_change_the_representation_once_learned(self):
        torch.manual_seed(2)
        model = F5SequenceResidual(self.weight, pooling='mean').eval()
        with torch.no_grad():
            model.type_embedding.weight[2:].normal_()
            base = model.representation(self.candidate, self.history, self.types)[1]
            other = torch.tensor([[5, 4, 0, 0], [3, 0, 0, 0]])
            changed = model.representation(self.candidate, self.history, other)[1]
        self.assertFalse(torch.allclose(base, changed))

    def test_pad_and_unknown_categories_are_masked_and_empty_is_exactly_zero(self):
        torch.manual_seed(3)
        model = F5SequenceResidual(self.weight, pooling='attention').eval()
        with torch.no_grad():
            model.type_embedding.weight[2:].normal_()
            for parameter in model.head.parameters():
                parameter.normal_()
            # Category 1 is UNKNOWN: a history of only UNKNOWN is no history.
            unknown_only = torch.tensor([[1, 1, 0, 0]])
            value = model(torch.tensor([5]), unknown_only, torch.tensor([[2, 3, 0, 0]]))
            self.assertEqual(float(value.abs().max()), 0.0)
            # A fully padded history is also exactly zero, biases included.
            padded = torch.zeros(1, 4, dtype=torch.long)
            value = model(torch.tensor([5]), padded, padded)
            self.assertEqual(float(value.abs().max()), 0.0)

    def test_attention_is_candidate_conditioned(self):
        torch.manual_seed(4)
        model = F5SequenceResidual(self.weight, pooling='attention').eval()
        with torch.no_grad():
            history = torch.tensor([[7, 8, 9, 0]])
            types = torch.tensor([[2, 3, 4, 0]])
            first = model.representation(torch.tensor([5]), history, types)[1]
            second = model.representation(torch.tensor([6]), history, types)[1]
        self.assertFalse(torch.allclose(first, second))

    def test_both_poolings_are_permutation_invariant(self):
        # Neither operator carries positional or time encoding, so permuting the
        # same history tokens cannot change the output. This is why an
        # order-shuffle ablation would measure nothing on this architecture, and
        # the property is asserted here rather than only described in prose.
        torch.manual_seed(5)
        history = torch.tensor([[7, 8, 9, 5]])
        types = torch.tensor([[2, 3, 4, 5]])
        permutation = torch.tensor([3, 1, 0, 2])
        for pooling in ('mean', 'attention'):
            model = F5SequenceResidual(self.weight, pooling=pooling).eval()
            with torch.no_grad():
                model.type_embedding.weight[2:].normal_()
                for parameter in model.head.parameters():
                    parameter.normal_()
                original = model(torch.tensor([6]), history, types)
                shuffled = model(
                    torch.tensor([6]), history[:, permutation], types[:, permutation]
                )
            torch.testing.assert_close(original, shuffled, msg=f'{pooling} pooling')

    def test_assert_neutral_rejects_a_perturbed_reserved_row(self):
        model = F5SequenceResidual(self.weight, pooling='mean')
        model.assert_neutral()
        with torch.no_grad():
            model.type_embedding.weight[1].fill_(0.5)
        with self.assertRaises(AssertionError):
            model.assert_neutral()


class ZeroControlTests(unittest.TestCase):
    def test_zero_control_emits_zero_and_blocks_gradients(self):
        weight = embedding_fixture()
        torch.manual_seed(6)
        model = ZeroF5SequenceResidual(weight, pooling='attention')
        with torch.no_grad():
            # Deliberately inject signal everywhere the control could leak from.
            model.type_embedding.weight.normal_()
            for parameter in model.head.parameters():
                parameter.normal_()
        candidate = torch.tensor([5, 6])
        history = torch.tensor([[7, 8, 0, 0], [9, 5, 6, 0]])
        types = torch.tensor([[2, 3, 0, 0], [4, 5, 2, 0]])
        value = model(candidate, history, types)
        self.assertEqual(float(value.detach().abs().max()), 0.0)
        value.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertEqual(
                    int(torch.count_nonzero(parameter.grad)), 0, f'{name} received gradient'
                )


if __name__ == '__main__':
    unittest.main()
