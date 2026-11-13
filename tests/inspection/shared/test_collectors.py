"""Test primitive collectors independently of any model family."""
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from attention_quantization.inspection.collectors.tensor import TensorCollector
from attention_quantization.inspection.collectors.gradient import GradientCollector
from attention_quantization.inspection.core.targets import CaptureTarget
from attention_quantization.inspection.core.composite import CompositeCollector


class CollectorTests(unittest.TestCase):
    def test_input_and_output_capture_preserve_execution(self):
        """Pre/post hooks observe exact tensors without replacing module values."""
        module = nn.Linear(2, 2)
        source = torch.tensor([[1., 2.]])
        expected = module(source)
        targets = [CaptureTarget("input", 0, module, lambda args: args[0], True),
                   CaptureTarget("output", 0, module, lambda output: output)]
        received = []
        collector = TensorCollector(targets)
        try:
            collector.start(lambda target, tensor: received.append((target.name, tensor)))
            actual = module(source)
        finally:
            collector.close()
        self.assertIs(received[0][1], source)
        self.assertIs(received[1][1], actual)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(module._forward_hooks)
        self.assertFalse(module._forward_pre_hooks)

    def test_partial_hook_registration_rolls_back(self):
        """A later registration failure removes hooks already attached."""
        module = nn.Identity()
        broken = Mock()
        broken.register_forward_hook.side_effect = RuntimeError("registration failed")
        collector = TensorCollector([
            CaptureTarget("ok", 0, module, lambda value: value),
            CaptureTarget("broken", 1, broken, lambda value: value),
        ])
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            collector.start(Mock())
        self.assertFalse(module._forward_hooks)
        collector.close()

    def test_composite_rolls_back_in_reverse_order(self):
        """A failing child causes cleanup of all children in reverse order."""
        events = []
        first, second = Mock(), Mock()
        first.close.side_effect = lambda: events.append("first")
        second.close.side_effect = lambda: events.append("second")
        second.start.side_effect = RuntimeError("child failed")
        with self.assertRaisesRegex(RuntimeError, "child failed"):
            CompositeCollector([first, second]).start(Mock())
        self.assertEqual(events, ["second", "first"])

    def test_gradient_validation(self):
        """Reject nonscalar objectives, detached captures, and empty targets."""
        collector = GradientCollector()
        source = torch.ones(2, requires_grad=True)
        with self.assertRaises(ValueError):
            collector.compute(source, [source])
        for tensors in ([], [source.detach()]):
            with self.subTest(count=len(tensors)), self.assertRaises(RuntimeError):
                collector.compute(source.sum(), tensors)

    def test_disconnected_gradient_is_an_error(self):
        """Unused tensors fail explicitly instead of yielding misleading zero scores."""
        source = torch.ones(2, requires_grad=True)
        unrelated = torch.ones(2, requires_grad=True)
        with self.assertRaises(RuntimeError):
            GradientCollector().compute(source.sum(), [unrelated])
