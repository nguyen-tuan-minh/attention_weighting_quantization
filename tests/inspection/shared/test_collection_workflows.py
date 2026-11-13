"""Verify collection extension and per-forward accumulation independently of LLaVA."""
import unittest
import torch
from torch import nn

from attention_quantization.inspection.core.inspector import ModelInspector
from attention_quantization.inspection.core.targets import CaptureTarget
from attention_quantization.inspection.core.results import CollectionCapture
from attention_quantization.inspection.collectors.tensor import TensorCollector
from attention_quantization.inspection.collectors.gradient import GradientCollector
from attention_quantization.inspection.collections.composites import Collection, GradientImportanceCollector


class SumCollector(Collection):
    """Example custom workflow unknown to CaptureSession."""

    default_label = "sum"

    def required_targets(self):
        """Request the fixture model's output tensor."""
        return ["output"]

    def measure_tensor(self, target, tensor):
        """Return a scalar sum without retaining the source activation."""
        return self.label, tensor.sum()


class FixtureInspector(ModelInspector):
    """Bind a single linear model to the generic session API."""

    def targets(self):
        """Describe the fixture output at logical layer zero."""
        return [CaptureTarget("output", 0, self.model, lambda output: output)]

    def tensor_collector(self, targets):
        """Construct the generic tensor primitive."""
        return TensorCollector(targets)

    def gradient_collector(self):
        """Construct the generic derivative primitive."""
        return GradientCollector()



class CollectionWorkflowTests(unittest.TestCase):
    def test_custom_collector_output_declaration(self):
        """A custom collector inherits output naming without inspector/session changes."""
        model = nn.Linear(2, 2)
        inspector = FixtureInspector(model)
        collector = SumCollector(name="analysis", label="custom_total")
        with inspector.capture(targets=[collector]) as session:
            output = model(torch.ones(1, 2))
        measurements = session.results[0].collections[collector.name].measurements
        self.assertEqual(set(measurements), {(0, "custom_total")})
        self.assertEqual(collector.labels, ("custom_total",))
        torch.testing.assert_close(measurements[collector.measurement_key(0)][0], output.sum())

    def test_new_collection_requires_no_session_changes(self):
        """A custom dependency and reducer work through the existing public API."""
        model = nn.Linear(2, 2)
        inspector = FixtureInspector(model)
        inputs = torch.ones(1, 2)
        with inspector.capture(targets=[SumCollector], scoped=True) as session:
            with session.forward():
                output = model(inputs)
        torch.testing.assert_close(session.results[0].collections["SumCollector"].measurements[(0, "sum")][0], output.sum())
        self.assertEqual(session.results[0].tensors[(0, "output")], [None])

    def test_accumulators_are_isolated_by_forward_and_call(self):
        """Repeated objectives cannot mix two forwards or two invocations of a layer."""
        collector = GradientImportanceCollector(reduction="sum_squared", accumulate=True)
        first, second = CollectionCapture(collector.snapshot()), CollectionCapture(collector.snapshot())
        key = (0, "attention.value")
        for record, call, scale in [(first, 0, 1), (first, 1, 2), (first, 0, 3), (second, 0, 4)]:
            collector.process_gradient(record, key, call, torch.full((1, 2, 3), float(scale)),
                                       lambda value: value, lambda value: value)
        torch.testing.assert_close(first.measurements[(0, "gradient_importance")][0][0], torch.full((2,), 30.))
        torch.testing.assert_close(first.measurements[(0, "gradient_importance")][1][0], torch.full((2,), 12.))
        torch.testing.assert_close(second.measurements[(0, "gradient_importance")][0][0], torch.full((2,), 48.))
