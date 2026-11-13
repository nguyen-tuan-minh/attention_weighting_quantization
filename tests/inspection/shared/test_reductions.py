"""Numerical checks with explicit attention probabilities and expected results."""
import unittest
import torch

from attention_quantization.inspection.collections.attention import iga, attention_mass, mask_for


class ReductionTests(unittest.TestCase):
    def setUp(self):
        """Prepare two heads with distinct query rows and normalized key weights."""
        self.weights = torch.tensor([[
            [[.1, .2, .7], [.3, .4, .3]],
            [[.5, .2, .3], [.1, .6, .3]],
        ]])

    def test_iga_averages_selected_queries_and_heads(self):
        """Image-key means use selected queries and every head exactly once."""
        result = iga(self.weights, [False, True], [True, False, True])
        torch.testing.assert_close(result[0], torch.tensor([.2, .3]))

    def test_iga_supports_ragged_per_sample_selections(self):
        """Batch masks may select different numbers of image tokens and queries."""
        weights = self.weights.expand(2, -1, -1, -1)
        result = iga(weights, [[True, False], [False, True]],
                     [[True, False, False], [True, False, True]])
        torch.testing.assert_close(result[0], torch.tensor([.3]))
        torch.testing.assert_close(result[1], torch.tensor([.2, .3]))

    def test_mass_preserves_total_attention_and_row_count(self):
        """A complete key partition sums to the number of selected head/query rows."""
        result = attention_mass(self.weights, [True, True],
                                [True, False, False], [False, True, True])[0]
        torch.testing.assert_close(result["image"], torch.tensor(1.))
        torch.testing.assert_close(result["text"], torch.tensor(3.))
        self.assertEqual(result["query_rows"], 4)

    def test_invalid_masks_are_rejected(self):
        """Reject wrong lengths, wrong batch sizes, and empty sample selections."""
        for mask in ([True], [[True, False], [True, False]], [False, False]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                mask_for(mask, self.weights, -2)

    def test_overlapping_modalities_are_rejected(self):
        """Mass cannot count the same key in both modalities."""
        with self.assertRaisesRegex(ValueError, "disjoint"):
            attention_mass(self.weights, [True, True], [True, False, False], [True, True, True])

    def test_iga_requires_attention_layout(self):
        """IGA requires separate batch, head, query, and key dimensions."""
        with self.assertRaises(ValueError):
            iga(torch.ones(2, 3, 4), [True] * 3, [True] * 4)
