"""Regression checks for investigation scoring without CLI/model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from attention_quantization.inspection import LlavaInspector, GradientImportanceCollector, TensorCapture
if __package__:
    from .test_inspector import Model
else:
    from test_inspector import Model


def load_scoring_helpers():
    """Load actual script functions without importing optional plotting/data packages."""
    path = Path(__file__).resolve().parents[3] / "scripts/investigate/investigate_grad_important.py"
    source = ast.parse(path.read_text())
    names = {"score_candidate_token_gradients", "score_assistant_token_losses"}
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    namespace = dict(torch=torch, F=F, LlavaInspector=LlavaInspector,
                     GradientImportanceCollector=GradientImportanceCollector)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class CausalModel(Model):
    def __init__(self):
        """Add a tiny vocabulary head to the local decoder fixture."""
        super().__init__()
        self.head = nn.Linear(4, 7)

    def forward(self, inputs_embeds, **kwargs):
        """Accept investigation model kwargs and return vocabulary logits."""
        return SimpleNamespace(logits=self.head(super().forward(inputs_embeds)))


class InvestigationCaptureTests(unittest.TestCase):
    def setUp(self):
        """Create deterministic CPU scoring inputs with disjoint modality masks."""
        torch.manual_seed(23)
        self.model = CausalModel()
        self.x = torch.randn(1, 3, 4)
        self.image = torch.tensor([True, False, False])
        self.text = ~self.image
        self.helpers = load_scoring_helpers()

    def assert_hooks_removed(self):
        """Check that generation after scoring cannot retain investigation hooks."""
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in self.model.modules()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_candidate_matches_direct_autograd_for_each_capture_point(self):
        """All three CLI activation points preserve candidate squared-gradient scores."""
        for target in ("block.input", "input_norm.output", "attention.value"):
            with self.subTest(target=target):
                with LlavaInspector(self.model).capture(targets=[TensorCapture(target)], storage="live") as reference:
                    logits = self.model(self.x.detach().requires_grad_(True)).logits
                    loss = F.cross_entropy(logits[:, -1].float(), torch.tensor([2]))
                    tensors = [reference.results[0].tensors[(layer, target)][0] for layer in range(2)]
                    expected = [g.square().sum(-1)[0] for g in torch.autograd.grad(loss, tensors)]
                image, text, actual_loss = self.helpers["score_candidate_token_gradients"](
                    self.model, target, self.x, torch.ones(1, 3, dtype=torch.bool),
                    self.image, self.text, torch.tensor(2))
                self.assertAlmostEqual(actual_loss, loss.item(), places=6)
                for layer, score in enumerate(expected):
                    torch.testing.assert_close(image[layer], score[self.image])
                    torch.testing.assert_close(text[layer], score[self.text])
                self.assert_hooks_removed()

    def test_per_token_accumulation_without_full_tensor_retention(self):
        """Reduced-only capture sums separate CE gradients and clears hooks on exit."""
        labels = torch.tensor([-100, 2, 4])
        with LlavaInspector(self.model).capture(targets=[TensorCapture("block.input")], storage="live") as reference:
            logits = self.model(self.x.detach().requires_grad_(True)).logits
            tensors = [reference.results[0].tensors[(layer, "block.input")][0] for layer in range(2)]
            expected = [torch.zeros(3) for _ in tensors]
            for position in (1, 2):
                loss = F.cross_entropy(logits[:, position - 1], labels[position:position + 1])
                gradients = torch.autograd.grad(loss, tensors, retain_graph=position == 1)
                for scores, gradient in zip(expected, gradients):
                    scores.add_(gradient.square().sum(-1)[0])
        collector = GradientImportanceCollector(gradient_target="block.input", reduction="sum_squared",
                                                accumulate=True, label="token_energy")
        with LlavaInspector(self.model).capture(targets=[collector], storage="cpu", scoped=True,
                                                retain_tensors=False, retain_gradients=False) as capture:
            with capture.forward():
                logits = self.model(self.x.detach().requires_grad_(True)).logits
                image, text, losses = self.helpers["score_assistant_token_losses"](
                    logits, labels, self.image, self.text, capture, collector)
            self.assertEqual(len(losses), 2)
            self.assertEqual(capture.results[0].gradients, {})
            self.assertTrue(all(value is None for calls in capture.results[0].tensors.values() for value in calls))
            self.model(self.x)
            self.assertEqual(len(capture.results), 1)
        for layer, score in enumerate(expected):
            torch.testing.assert_close(image[layer], score[self.image])
            torch.testing.assert_close(text[layer], score[self.text])
        self.assert_hooks_removed()

    def test_candidate_failure_removes_hooks(self):
        """An invalid vocabulary target cannot leave scoring hooks installed."""
        with self.assertRaises((IndexError, RuntimeError)):
            self.helpers["score_candidate_token_gradients"](
                self.model, "attention.value", self.x, torch.ones(1, 3, dtype=torch.bool),
                self.image, self.text, torch.tensor(99))
        self.assert_hooks_removed()
