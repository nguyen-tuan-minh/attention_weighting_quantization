"""CPU-only behavioral checks; no model downloads required."""
import unittest
import torch
from torch import nn
from attention_quantization.inspection import LlavaInspector


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        weights = (q @ k.transpose(-1, -2)).softmax(-1).unsqueeze(1)
        return weights[:, 0] @ v, weights


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = Attention()

    def forward(self, x):
        return x + self.self_attn(self.input_layernorm(x))[0]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class InspectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Model()
        self.inspector = LlavaInspector(self.model)
        self.x = torch.randn(2, 3, 4, requires_grad=True)

    def test_capture_forwards_and_cleanup(self):
        with self.inspector.capture(targets=["attention.key", "input_norm.output"]) as capture:
            self.model(self.x)
            self.model(self.x)
        self.assertEqual(len(capture.results), 2)
        tensor = capture.results[0].tensors[(0, "attention.key")][0]
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_gradient_matches_autograd(self):
        with self.inspector.capture(targets=["attention.value"], gradients=True) as capture:
            output = self.model(self.x)
            tensor = capture.results[0].tensors[(0, "attention.value")][0]
            expected = torch.autograd.grad(output.sum(), tensor, retain_graph=True)[0]
            capture.compute_gradients(output.sum())
            actual = capture.results[0].gradients[(0, "attention.value")][0]
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_iga_and_overlap(self):
        with self.inspector.capture(collection="overlap", query_mask=[False, True, True],
                                    image_mask=[True, False, False], storage="cpu") as capture:
            output = self.model(self.x)
            capture.compute_gradients(output.square().mean())
        record = capture.results[0]
        self.assertEqual(record.measurements[(0, "overlap")][0], [1.0, 1.0])
        self.assertEqual(record.measurements[(0, "iga")][0][0].shape, (1,))

    def test_cleanup_on_exception_and_unsupported_targets(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.inspector.capture(targets=["missing"])
        with self.assertRaisesRegex(RuntimeError, "example"):
            with self.inspector.capture(targets=["block.output"]):
                raise RuntimeError("example")
        with self.inspector.capture(targets=["block.output"]):
            self.model(self.x)


if __name__ == "__main__":
    unittest.main()
