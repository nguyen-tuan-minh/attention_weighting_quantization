"""Vision capture checks on a small CPU encoder without external checkpoints."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from attention_quantization.inspection import LlavaVisionInspector, TensorCapture


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(4)
        self.layer_norm2 = nn.LayerNorm(4)
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, nn.Linear(4, 4))
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        normalized = self.layer_norm1(x)
        x = x + self.self_attn.out_proj(self.self_attn.v_proj(normalized))
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="clip_vision_model")
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([VisionBlock(), VisionBlock()])

    def forward(self, x):
        for block in self.vision_model.encoder.layers:
            x = block(x)
        return x


class VisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.encoder = Encoder()
        tower = SimpleNamespace(vision_tower=self.encoder)
        self.inspector = LlavaVisionInspector(SimpleNamespace(get_vision_tower=lambda: tower))

    def assert_clean(self):
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in self.encoder.modules()))

    def test_gradients_match_direct_autograd(self):
        x = torch.randn(1, 5, 4, requires_grad=True)
        with self.inspector.capture(targets=[TensorCapture("attention.value", gradients=True)]) as capture:
            output = self.encoder(x)
            value = capture.results[0].tensors[(0, "attention.value")][0]
            expected, = torch.autograd.grad(output.square().sum(), value, retain_graph=True)
            capture.compute_gradients(output.square().sum())
            actual = capture.results[0].gradients[(0, "attention.value")][0]
            torch.testing.assert_close(actual, expected)
        self.assert_clean()
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))

    def test_linear_sink_matches_activation_gram_matrix(self):
        name = "vision_model.encoder.layers.0.self_attn.v_proj"
        index = list(self.inspector.linear_modules()).index(name)
        x = torch.randn(1, 5, 4)
        expected_rows = self.encoder.vision_model.encoder.layers[0].layer_norm1(x).reshape(-1, 4)
        matrices = []
        def sink(_forward, _layer, _target, _call, value):
            rows = value.detach().reshape(-1, 4)
            matrices.append(rows.T @ rows)
        with self.inspector.capture(targets=[TensorCapture("linear.input")], layers=[index],
                                    storage="live", sink=sink, retain_tensors=False):
            self.encoder(x)
        torch.testing.assert_close(matrices[0], expected_rows.T @ expected_rows)
        self.assert_clean()

    def test_exception_removes_hooks(self):
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with self.inspector.capture(targets=[TensorCapture("block.input")]):
                self.encoder(torch.randn(1, 5, 4))
                raise RuntimeError("stop")
        self.assert_clean()
