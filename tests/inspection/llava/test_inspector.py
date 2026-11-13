"""CPU-only behavioral checks; no model downloads required."""
import unittest
import tempfile
import torch
from torch import nn
from attention_quantization.inspection import LlavaInspector


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
        with self.inspector.capture(targets=["attention.key", "input_norm.output"]) as capture:
            self.model(self.x)
            self.model(self.x)
        self.assertEqual(len(capture.results), 2)
        tensor = capture.results[0].tensors[(0, "attention.key")][0]
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_gradient_matches_autograd(self):
        """Captured gradients equal direct autograd without parameter accumulation."""
        with self.inspector.capture(targets=["attention.value"], gradients=True) as capture:
            output = self.model(self.x)
            tensor = capture.results[0].tensors[(0, "attention.value")][0]
            expected = torch.autograd.grad(output.sum(), tensor, retain_graph=True)[0]
            capture.compute_gradients(output.sum())
            actual = capture.results[0].gradients[(0, "attention.value")][0]
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_iga_and_overlap(self):
        """One-image-token overlap is complete for every sample and selected layer."""
        with self.inspector.capture(collection="overlap", query_mask=[False, True, True],
                                    image_mask=[True, False, False], storage="cpu") as capture:
            output = self.model(self.x)
            capture.compute_gradients(output.square().mean())
        record = capture.results[0]
        self.assertEqual(record.measurements[(0, "overlap")][0], [1.0, 1.0])
        self.assertEqual(record.measurements[(0, "iga")][0][0].shape, (1,))

    def test_cleanup_on_exception_and_unsupported_targets(self):
        """Invalid requests and failed scopes allow subsequent capture sessions."""
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.inspector.capture(targets=["missing"])
        with self.assertRaisesRegex(RuntimeError, "example"):
            with self.inspector.capture(targets=["block.output"]):
                raise RuntimeError("example")
        with self.inspector.capture(targets=["block.output"]):
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
        with self.inspector.capture(targets=["attention.key"] * 2, layers=[1]) as capture:
            self.model(self.x)
        self.assertEqual(set(capture.results[0].tensors), {(1, "attention.key")})
        self.assertEqual(len(capture.results[0].tensors[(1, "attention.key")]), 1)

    def test_invalid_requests_fail_without_attaching_hooks(self):
        """Invalid layers, collections, required options, and overlap fractions fail early."""
        requests = [dict(targets=["block.output"], layers=[-1]),
                    dict(targets=["block.output"], layers=[2]),
                    dict(targets=["block.output"], layers=[]),
                    dict(collection="missing"), dict(collection="iga"),
                    dict(collection="attention_mass", query_mask=[True] * 3,
                         image_mask=[True, False, False]),
                    dict(collection="overlap", query_mask=[True] * 3,
                         image_mask=[True, False, False], top_fraction=0)]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.inspector.capture(**request)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in self.model.modules()))

    def test_capture_does_not_change_model_output(self):
        """Raw attention capture leaves forward numerics unchanged."""
        expected = self.model(self.x)
        with self.inspector.capture(targets=["attention.weights", "block.input"]) as capture:
            actual = self.model(self.x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(capture.results[0].tensors[(0, "block.input")][0], self.x)

    def test_disable_clear_and_resume(self):
        """Disabled forwards are skipped; clearing releases results and resets indices."""
        with self.inspector.capture(targets=["block.output"]) as capture:
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
        with self.inspector.capture(targets=["block.output"]) as outer:
            with self.assertRaises(RuntimeError):
                with self.inspector.capture(targets=["block.input"]):
                    pass
            self.model(self.x)
        self.assertEqual(len(outer.results), 1)

    def test_repeated_module_invocations_are_preserved(self):
        """A shared decoder block keeps each module call within the same forward."""
        self.model.model.layers[1] = self.model.model.layers[0]
        with self.inspector.capture(targets=["block.output"], layers=[0]) as capture:
            self.model(self.x)
        self.assertEqual(len(capture.results[0].tensors[(0, "block.output")]), 2)

    def test_multiple_gradient_objectives(self):
        """Retaining a graph supports distinct objectives on the same forward."""
        with self.inspector.capture(targets=["attention.value"], layers=[0], gradients=True) as capture:
            output = self.model(self.x)
            capture.compute_gradients(output.sum(), retain_graph=True)
            capture.compute_gradients(2 * output.sum())
            gradients = capture.results[0].gradients[(0, "attention.value")]
            self.assertEqual(len(gradients), 2)
            torch.testing.assert_close(gradients[1], 2 * gradients[0])

    def test_gradient_selection_uses_requested_forward(self):
        """Differentiation can target an earlier forward without filling a later result."""
        with self.inspector.capture(targets=["attention.value"], gradients=True) as capture:
            first = self.model(self.x)
            self.model(self.x + 1)
            capture.compute_gradients(first.sum(), forward=0)
            self.assertTrue(capture.results[0].gradients)
            self.assertFalse(capture.results[1].gradients)

    def test_gradient_scope_and_no_grad_validation(self):
        """Gradient requests outside a live scope or under no_grad fail clearly."""
        with self.inspector.capture(targets=["attention.value"], gradients=True) as capture:
            with torch.no_grad():
                output = self.model(self.x)
            with self.assertRaises(RuntimeError):
                capture.compute_gradients(output.sum())
        with self.assertRaises(RuntimeError):
            capture.compute_gradients(output.sum())

    def test_importance_matches_explicit_gradient_norm(self):
        """High-level importance equals hidden-dimension L2 gradient norms."""
        with self.inspector.capture(collection="gradient_importance", layers=[0]) as capture:
            output = self.model(self.x)
            value = capture.results[0].tensors[(0, "attention.value")][0]
            expected = torch.autograd.grad(output.sum(), value, retain_graph=True)[0].norm(dim=-1)
            capture.compute_gradients(output.sum())
            actual = capture.results[0].measurements[(0, "gradient_importance")][0]
            torch.testing.assert_close(torch.stack(actual), expected)

    def test_attention_mass_collection_matches_raw_weights(self):
        """Public aggregate mass matches an independently summed raw attention capture."""
        with self.inspector.capture(targets=["attention.weights"], layers=[0]) as raw:
            self.model(self.x)
        weights = raw.results[0].tensors[(0, "attention.weights")][0]
        with self.inspector.capture(collection="attention_mass", layers=[0],
                                    query_mask=[False, True, True], image_mask=[True, False, False],
                                    text_mask=[False, True, True]) as capture:
            self.model(self.x)
        values = capture.results[0].measurements[(0, "attention_mass")][0]
        for sample, result in zip(weights, values):
            torch.testing.assert_close(result["image"], sample[:, 1:, 0].sum())
            torch.testing.assert_close(result["text"], sample[:, 1:, 1:].sum())

    def test_disk_overlap_round_trip(self):
        """Overlap reads spilled IGA vectors and preserves final artifact files."""
        with tempfile.TemporaryDirectory() as directory:
            with self.inspector.capture(collection="overlap", layers=[0], storage="disk",
                                        directory=directory, query_mask=[False, True, True],
                                        image_mask=[True, False, False]) as capture:
                output = self.model(self.x)
                capture.compute_gradients(output.sum())
            record = capture.results[0]
            path = record.measurements[(0, "iga")][0][0]
            self.assertTrue(path.exists())
            self.assertEqual(torch.load(path, weights_only=True).shape, (1,))
            self.assertEqual(record.measurements[(0, "overlap")][0], [1., 1.])

    def test_forward_failure_removes_all_hooks(self):
        """An error during tensor extraction cleans hooks and releases scope ownership."""
        self.model.model.layers[0].self_attn = nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            with self.inspector.capture(targets=["attention.weights"]):
                self.model(self.x)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in self.model.modules()))


if __name__ == "__main__":
    unittest.main()
