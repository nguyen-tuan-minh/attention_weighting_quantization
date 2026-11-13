"""CPU-only behavioral checks; no model downloads required."""
import unittest
import tempfile
import torch
from torch import nn
from attention_quantization.inspection import (
    LlavaInspector, TensorCapture, ValueCollector, AttentionWeightsCollector,
    IGACollector, AttentionMassCollector, GradientImportanceCollector, OverlapCollector,
)


class Attention(nn.Module):
    def __init__(self):
        """Create small query/key/value projections with no external checkpoint."""
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)

    def forward(self, x):
        """Return attended values and [batch, 1, query, key] probabilities."""
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        weights = (q @ k.transpose(-1, -2)).softmax(-1).unsqueeze(1)
        return weights[:, 0] @ v, weights


class Block(nn.Module):
    def __init__(self):
        """Create a normalization and attention residual block."""
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = Attention()

    def forward(self, x):
        """Apply normalized self-attention while preserving the residual path."""
        return x + self.self_attn(self.input_layernorm(x))[0]


class Model(nn.Module):
    def __init__(self):
        """Expose two blocks at the decoder location expected by LLaVA."""
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        """Execute both blocks and return [batch, sequence, hidden] outputs."""
        for layer in self.model.layers:
            x = layer(x)
        return x


class InspectionTests(unittest.TestCase):
    def setUp(self):
        """Create a deterministic two-sample CPU model and gradient-enabled input."""
        torch.manual_seed(7)
        self.model = Model()
        self.inspector = LlavaInspector(self.model)
        self.x = torch.randn(2, 3, 4, requires_grad=True)

    def test_capture_forwards_and_cleanup(self):
        """Forwards stay distinct and detached captures survive hook cleanup."""
        with self.inspector.capture(targets=[TensorCapture(["attention.key", "input_norm.output"])]) as capture:
            self.model(self.x)
            self.model(self.x)
        self.assertEqual(len(capture.results), 2)
        tensor = capture.results[0].tensors[(0, "attention.key")][0]
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_gradient_matches_autograd(self):
        """Captured gradients equal direct autograd without parameter accumulation."""
        with self.inspector.capture(targets=[ValueCollector(gradients=True)]) as capture:
            output = self.model(self.x)
            tensor = capture.results[0].tensors[(0, "attention.value")][0]
            expected = torch.autograd.grad(output.sum(), tensor, retain_graph=True)[0]
            capture.compute_gradients(output.sum())
            actual = capture.results[0].gradients[(0, "attention.value")][0]
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_iga_and_overlap(self):
        """One-image-token overlap is complete for every sample and selected layer."""
        with self.inspector.capture(targets=[OverlapCollector(query_mask=[False, True, True],
                                                              image_mask=[True, False, False])], storage="cpu") as capture:
            output = self.model(self.x)
            capture.compute_gradients(output.square().mean())
        record = capture.results[0]
        self.assertEqual(record.collections["OverlapCollector"].measurements[(0, "overlap")][0], [1.0, 1.0])
        self.assertEqual(record.collections["OverlapCollector"].measurements[(0, "iga")][0][0].shape, (1,))


    def test_cleanup_on_exception_and_unsupported_targets(self):
        """Invalid requests and failed scopes allow subsequent capture sessions."""
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.inspector.capture(targets=[TensorCapture(["missing"])])
        with self.assertRaisesRegex(RuntimeError, "example"):
            with self.inspector.capture(targets=[TensorCapture(["block.output"])]):
                raise RuntimeError("example")
        with self.inspector.capture(targets=[TensorCapture(["block.output"])]):
            self.model(self.x)

    def test_target_discovery_and_invalid_model(self):
        """Expose only targets supported by this model and reject missing decoders."""
        available = self.inspector.capture_targets()
        self.assertEqual(available["attention.query"], [0, 1])
        self.assertNotIn("post_attention_norm.output", available)
        with self.assertRaises(ValueError):
            LlavaInspector(nn.Identity()).capture_targets()

    def test_layer_selection_and_duplicate_targets(self):
        """Repeated requests share one capture and excluded layers are absent."""
        with self.inspector.capture(targets=[TensorCapture("attention.key", name="first"), TensorCapture("attention.key", name="second")], layers=[1]) as capture:
            self.model(self.x)
        self.assertEqual(set(capture.results[0].tensors), {(1, "attention.key")})
        self.assertEqual(len(capture.results[0].tensors[(1, "attention.key")]), 1)

    def test_invalid_requests_fail_without_attaching_hooks(self):
        """Invalid layers, collections, required options, and overlap fractions fail early."""
        requests = [dict(targets=[TensorCapture("block.output")], layers=[-1]),
                    dict(targets=[TensorCapture("block.output")], layers=[2]),
                    dict(targets=[TensorCapture("block.output")], layers=[]),
                    dict(targets=[])]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.inspector.capture(**request)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in self.model.modules()))

    def test_capture_does_not_change_model_output(self):
        """Raw attention capture leaves forward numerics unchanged."""
        expected = self.model(self.x)
        with self.inspector.capture(targets=[TensorCapture(["attention.weights", "block.input"])]) as capture:
            actual = self.model(self.x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(capture.results[0].tensors[(0, "block.input")][0], self.x)

    def test_disable_clear_and_resume(self):
        """Disabled forwards are skipped; clearing releases results and resets indices."""
        with self.inspector.capture(targets=[TensorCapture(["block.output"])]) as capture:
            self.model(self.x)
            capture.enabled = False
            self.model(self.x)
            self.assertEqual(len(capture.results), 1)
            capture.clear()
            capture.enabled = True
            self.model(self.x)
            self.assertEqual(len(capture.results), 1)
            self.assertEqual(capture.results[0].index, 0)

    def test_nested_session_rejection_preserves_outer_session(self):
        """A rejected nested session must not remove the active session's hooks."""
        with self.inspector.capture(targets=[TensorCapture(["block.output"])]) as outer:
            with self.assertRaises(RuntimeError):
                with self.inspector.capture(targets=[TensorCapture(["block.input"])]):
                    pass
            self.model(self.x)
        self.assertEqual(len(outer.results), 1)

    def test_repeated_module_invocations_are_preserved(self):
        """A shared decoder block keeps each module call within the same forward."""
        self.model.model.layers[1] = self.model.model.layers[0]
        with self.inspector.capture(targets=[TensorCapture(["block.output"])], layers=[0]) as capture:
            self.model(self.x)
        self.assertEqual(len(capture.results[0].tensors[(0, "block.output")]), 2)

    def test_multiple_gradient_objectives(self):
        """Retaining a graph supports distinct objectives on the same forward."""
        with self.inspector.capture(targets=[ValueCollector(gradients=True)], layers=[0]) as capture:
            output = self.model(self.x)
            capture.compute_gradients(output.sum(), retain_graph=True)
            capture.compute_gradients(2 * output.sum())
            gradients = capture.results[0].gradients[(0, "attention.value")]
            self.assertEqual(len(gradients), 2)
            torch.testing.assert_close(gradients[1], 2 * gradients[0])

    def test_gradient_selection_uses_requested_forward(self):
        """Differentiation can target an earlier forward without filling a later result."""
        with self.inspector.capture(targets=[ValueCollector(gradients=True)]) as capture:
            first = self.model(self.x)
            self.model(self.x + 1)
            capture.compute_gradients(first.sum(), forward=0)
            self.assertTrue(capture.results[0].gradients)
            self.assertFalse(capture.results[1].gradients)

    def test_gradient_scope_and_no_grad_validation(self):
        """Gradient requests outside a live scope or under no_grad fail clearly."""
        with self.inspector.capture(targets=[ValueCollector(gradients=True)]) as capture:
            with torch.no_grad():
                output = self.model(self.x)
            with self.assertRaises(RuntimeError):
                capture.compute_gradients(output.sum())
        with self.assertRaises(RuntimeError):
            capture.compute_gradients(output.sum())

    def test_importance_matches_explicit_gradient_norm(self):
        """High-level importance equals hidden-dimension L2 gradient norms."""
        with self.inspector.capture(targets=[GradientImportanceCollector()], layers=[0]) as capture:
            output = self.model(self.x)
            value = capture.results[0].tensors[(0, "attention.value")][0]
            expected = torch.autograd.grad(output.sum(), value, retain_graph=True)[0].norm(dim=-1)
            capture.compute_gradients(output.sum())
            actual = capture.results[0].collections["GradientImportanceCollector"].measurements[(0, "gradient_importance")][0]
            torch.testing.assert_close(torch.stack(actual), expected)


    def test_attention_mass_collection_matches_raw_weights(self):
        """Public aggregate mass matches an independently summed raw attention capture."""
        with self.inspector.capture(targets=[TensorCapture(["attention.weights"])], layers=[0]) as raw:
            self.model(self.x)
        weights = raw.results[0].tensors[(0, "attention.weights")][0]
        with self.inspector.capture(targets=[AttentionMassCollector(query_mask=[False, True, True],
                                    image_mask=[True, False, False], text_mask=[False, True, True])], layers=[0]) as capture:
            self.model(self.x)
        values = capture.results[0].collections["AttentionMassCollector"].measurements[(0, "attention_mass")][0]
        for sample, result in zip(weights, values):
            torch.testing.assert_close(result["image"], sample[:, 1:, 0].sum())
            torch.testing.assert_close(result["text"], sample[:, 1:, 1:].sum())


    def test_disk_overlap_round_trip(self):
        """Overlap reads spilled IGA vectors and preserves final artifact files."""
        with tempfile.TemporaryDirectory() as directory:
            with self.inspector.capture(targets=[OverlapCollector(query_mask=[False, True, True],
                                        image_mask=[True, False, False])], layers=[0], storage="disk", directory=directory) as capture:
                output = self.model(self.x)
                capture.compute_gradients(output.sum())
            record = capture.results[0]
            path = record.collections["OverlapCollector"].measurements[(0, "iga")][0][0]
            self.assertTrue(path.exists())
            self.assertEqual(torch.load(path, weights_only=True).shape, (1,))
            self.assertEqual(record.collections["OverlapCollector"].measurements[(0, "overlap")][0], [1., 1.])


    def test_forward_failure_removes_all_hooks(self):
        """An error during tensor extraction cleans hooks and releases scope ownership."""
        self.model.model.layers[0].self_attn = nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            with self.inspector.capture(targets=[TensorCapture(["attention.weights"])]):
                self.model(self.x)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in self.model.modules()))


    def test_scoped_capture_skips_generation_and_updates_masks(self):
        """Only measured forwards are captured, using each scope's image mask."""
        collector = IGACollector(query_mask=[False, True, True], image_mask=[True, False, False])
        with self.inspector.capture(targets=[collector], scoped=True) as session:
            self.model(self.x)
            with session.forward():
                self.model(self.x)
            collector.query_mask = [True, True, True]
            collector.image_mask = [True, True, False]
            with session.forward():
                self.model(self.x)
        self.assertEqual(len(session.results), 2)
        self.assertEqual(session.results[0].collections["IGACollector"].measurements[(0, "iga")][0][0].shape, (1,))
        self.assertEqual(session.results[1].collections["IGACollector"].measurements[(0, "iga")][0][0].shape, (2,))


    def test_scoped_forward_validation_and_recovery(self):
        """Missing forwards and nested scopes fail without poisoning subsequent scopes."""
        with self.inspector.capture(targets=[TensorCapture(["block.output"])], scoped=True) as session:
            with self.assertRaises(RuntimeError):
                with session.forward():
                    pass
            with session.forward():
                with self.assertRaises(RuntimeError):
                    with session.forward():
                        pass
                self.model(self.x)

    def test_dtype_and_streaming_sink(self):
        """The sink receives typed CPU snapshots without retaining tensor results."""
        received = []
        with self.inspector.capture(targets=[TensorCapture(["attention.key"])], layers=[0], dtype=torch.float16,
                                    retain_tensors=False, sink=lambda *event: received.append(event)) as session:
            self.model(self.x)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][-1].dtype, torch.float16)
        self.assertIsNone(session.results[0].tensors[(0, "attention.key")][0])

    def test_attention_release_runs_after_consumers(self):
        """Release removes weights from module returns after IGA has consumed them."""
        observed = []
        with self.inspector.capture(targets=[IGACollector(query_mask=[True] * 3, image_mask=[True, False, False])],
                                    release_attention=True) as session:
            handle = self.model.model.layers[0].self_attn.register_forward_hook(
                lambda module, inputs, output: observed.append(output[1]))
            try:
                self.model(self.x)
            finally:
                handle.remove()
        self.assertEqual(observed, [None])
        self.assertTrue(session.results[0].collections["IGACollector"].measurements)


    def test_sum_squared_gradient_accumulation(self):
        """Accumulated squared scores equal the sum of two explicit gradient objectives."""
        with self.inspector.capture(targets=[GradientImportanceCollector(reduction="sum_squared", accumulate=True)], layers=[0]) as session:
            output = self.model(self.x)
            value = session.results[0].tensors[(0, "attention.value")][0]
            gradient = torch.autograd.grad(output.sum(), value, retain_graph=True)[0]
            session.compute_gradients(output.sum(), retain_graph=True)
            session.compute_gradients(2 * output.sum())
            actual = session.results[0].collections["GradientImportanceCollector"].measurements[(0, "gradient_importance")][0]
            torch.testing.assert_close(torch.stack(actual), 5 * gradient.square().sum(-1))

    def test_collector_classes_can_mix_with_instances(self):
        """Default-constructible classes and configured collectors share one session."""
        iga = IGACollector(query_mask=[False, True, True], image_mask=[True, False, False])
        with self.inspector.capture(targets=[AttentionWeightsCollector, ValueCollector, iga], layers=[0],
                                    release_attention=True) as session:
            self.model(self.x)
        record = session.results[0]
        weights = record.tensors[(0, "attention.weights")][0]
        expected = weights[0, :, 1:, :1].mean((0, 1))
        actual = record.collections[iga.name].measurements[(0, "iga")][0][0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(record.tensors[(0, "attention.weights")]), 1)
        self.assertEqual(len(record.tensors[(0, "attention.value")]), 1)

    def test_multiple_iga_collectors_keep_independent_masks(self):
        """Two analyses of one attention tensor produce separately named results."""
        first = IGACollector(query_mask=[True, False, False], image_mask=[True, False, False], name="first")
        second = IGACollector(query_mask=[False, True, True], image_mask=[False, True, True], name="second")
        with self.inspector.capture(targets=[AttentionWeightsCollector, first, second], layers=[0]) as session:
            self.model(self.x)
        record = session.results[0]
        weights = record.tensors[(0, "attention.weights")][0]
        torch.testing.assert_close(record.collections["first"].measurements[(0, "iga")][0][0],
                                   weights[0, :, :1, :1].mean((0, 1)))
        torch.testing.assert_close(record.collections["second"].measurements[(0, "iga")][0][0],
                                   weights[0, :, 1:, 1:].mean((0, 1)))
        self.assertEqual(len(record.tensors[(0, "attention.weights")]), 1)

    def test_gradient_requirements_are_inferred_and_shared(self):
        """Different reductions use the same derivative with independent accumulation."""
        norm = GradientImportanceCollector(name="norm")
        squared = GradientImportanceCollector(name="squared", reduction="sum_squared", accumulate=True)
        with self.inspector.capture(targets=[norm, squared], layers=[0]) as session:
            output = self.model(self.x)
            session.compute_gradients(output.sum(), retain_graph=True)
            session.compute_gradients(output.sum())
            record = session.results[0]
            gradients = record.gradients[(0, "attention.value")]
            self.assertEqual(len(gradients), 2)
            expected = gradients[0].square().sum(-1)
            torch.testing.assert_close(torch.stack(record.collections["norm"].measurements[(0, "gradient_importance")][0]), expected.sqrt())
            torch.testing.assert_close(torch.stack(record.collections["squared"].measurements[(0, "gradient_importance")][0]), 2 * expected)

    def test_forward_snapshot_preserves_gradient_selection(self):
        """Changing a collector mask after a forward cannot change that forward's gradients."""
        collector = GradientImportanceCollector(image_mask=[True, False, False])
        with self.inspector.capture(targets=[collector], layers=[0]) as session:
            output = self.model(self.x)
            collector.image_mask = [True, True, False]
            session.compute_gradients(output.sum())
        scores = session.results[0].collections[collector.name].measurements[(0, "gradient_importance")][0]
        self.assertEqual(scores[0].shape, (1,))

    def test_invalid_collector_definitions_fail_before_hooks(self):
        """Reject strings, missing collector configuration, and duplicate result names."""
        for targets in (["attention.value"], [object()], [IGACollector], [ValueCollector, ValueCollector()]):
            with self.subTest(targets=targets), self.assertRaises((TypeError, ValueError)):
                self.inspector.capture(targets=targets)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_masks_are_not_accepted_by_inspector_or_session(self):
        """Analysis parameters belong solely to collectors, with no legacy kwargs path."""
        with self.assertRaises(TypeError):
            self.inspector.capture(targets=[ValueCollector], image_mask=[True, False, False])
        with self.assertRaises(TypeError):
            self.inspector.capture(targets=[ValueCollector], collection="iga")
        with self.assertRaises(TypeError):
            self.inspector.capture(targets=[ValueCollector], gradients=True)
        with self.inspector.capture(targets=[ValueCollector], scoped=True) as session:
            with self.assertRaises(TypeError):
                session.forward(image_mask=[True, False, False])

    def test_collector_dependency_mutation_is_rejected(self):
        """Bound hooks cannot silently diverge from a changed gradient target."""
        collector = GradientImportanceCollector()
        with self.inspector.capture(targets=[collector]) as session:
            collector.gradient_target = "attention.key"
            with self.assertRaisesRegex(ValueError, "dependencies"):
                self.model(self.x)
            self.assertEqual(session.results, [])

    def test_collector_constructor_validation(self):
        """Reject invalid reductions, top fractions, and empty raw tensor requests."""
        with self.assertRaises(ValueError):
            GradientImportanceCollector(reduction="unknown")
        with self.assertRaises(ValueError):
            OverlapCollector(query_mask=[True], image_mask=[True], top_fraction=0)
        with self.assertRaises(ValueError):
            TensorCapture([])

    def test_custom_attention_labels(self):
        """IGA and modality totals publish configured keys without default aliases."""
        iga = IGACollector(query_mask=[False, True, True], image_mask=[True, False, False],
                           label="patch_scores")
        mass = AttentionMassCollector(query_mask=[False, True, True], image_mask=[True, False, False],
                                      text_mask=[False, True, True], label="modality_totals")
        with self.inspector.capture(targets=[AttentionWeightsCollector, iga, mass], layers=[0]) as session:
            self.model(self.x)
        record = session.results[0]
        weights = record.tensors[(0, "attention.weights")][0]
        iga_result = record.collections[iga.name].measurements
        mass_result = record.collections[mass.name].measurements
        self.assertEqual(set(iga_result), {(0, "patch_scores")})
        self.assertEqual(set(mass_result), {(0, "modality_totals")})
        torch.testing.assert_close(iga_result[iga.measurement_key(0)][0][0], weights[0, :, 1:, :1].mean((0, 1)))
        torch.testing.assert_close(mass_result[mass.measurement_key(0)][0][0]["image"], weights[0, :, 1:, 0].sum())

    def test_custom_importance_output_accumulates(self):
        """Renamed importance retains accumulation behavior and creates no default key."""
        collector = GradientImportanceCollector(label="token_energy", reduction="sum_squared", accumulate=True)
        with self.inspector.capture(targets=[collector], layers=[0]) as session:
            output = self.model(self.x)
            session.compute_gradients(output.sum(), retain_graph=True)
            session.compute_gradients(2 * output.sum())
        record = session.results[0]
        measurements = record.collections[collector.name].measurements
        self.assertEqual(set(measurements), {(0, "token_energy")})
        gradient = record.gradients[(0, "attention.value")][0]
        torch.testing.assert_close(torch.stack(measurements[collector.measurement_key(0)][0]),
                                   5 * gradient.square().sum(-1))

    def test_overlap_follows_all_component_labels(self):
        """The composite reads renamed IGA and writes renamed importance/overlap on disk."""
        collector = OverlapCollector(query_mask=[False, True, True], image_mask=[True, False, False],
                                     label="agreement", iga_label="attention_scores",
                                     importance_label="gradient_scores")
        with tempfile.TemporaryDirectory() as directory:
            with self.inspector.capture(targets=[collector], layers=[0], storage="disk", directory=directory) as session:
                output = self.model(self.x)
                session.compute_gradients(output.sum())
            measurements = session.results[0].collections[collector.name].measurements
            self.assertEqual(set(measurements), {(0, name) for name in collector.labels})
            self.assertEqual(set(collector.labels), {"attention_scores", "gradient_scores", "agreement"})
            self.assertEqual(measurements[collector.measurement_key(0)][0], [1., 1.])
            self.assertTrue(measurements[collector.iga.measurement_key(0)][0][0].exists())
            self.assertTrue(measurements[collector.importance.measurement_key(0)][0][0].exists())

    def test_labels_are_snapshotted_per_forward(self):
        """Renaming a component after execution cannot break that forward's dependency lookup."""
        collector = OverlapCollector(query_mask=[True] * 3, image_mask=[True, False, False])
        with self.inspector.capture(targets=[collector], layers=[0]) as session:
            output = self.model(self.x)
            collector.iga.label = "next_attention"
            collector.importance.label = "next_importance"
            collector.label = "next_overlap"
            session.compute_gradients(output.sum())
            next_output = self.model(self.x)
            session.compute_gradients(next_output.sum())
        first = session.results[0].collections[collector.name]
        second = session.results[1].collections[collector.name]
        self.assertEqual(set(first.measurements), {(0, name) for name in first.collector.labels})
        self.assertEqual(set(second.measurements), {(0, name) for name in second.collector.labels})
        self.assertTrue(set(first.measurements).isdisjoint(second.measurements))

    def test_invalid_labels_and_composite_collisions(self):
        """Reject malformed names and shared keys that would overwrite component results."""
        for label in ("", " ", 12, [], False):
            with self.subTest(label=label), self.assertRaises(ValueError):
                IGACollector(query_mask=[True], image_mask=[True], label=label)
        for names in (
            dict(label="same", iga_label="same"),
            dict(iga_label="same", importance_label="same"),
            dict(label="same", importance_label="same"),
        ):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "unique"):
                OverlapCollector(query_mask=[True], image_mask=[True], **names)
        collector = OverlapCollector(query_mask=[True], image_mask=[True])
        collector.iga.label = collector.label
        with self.assertRaisesRegex(ValueError, "unique"):
            self.inspector.capture(targets=[collector])



if __name__ == "__main__":
    unittest.main()
