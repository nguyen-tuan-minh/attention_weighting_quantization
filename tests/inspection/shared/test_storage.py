"""Verify storage ownership, graph retention, and caller-owned disk artifacts."""
import tempfile
import unittest
from pathlib import Path

import torch

from attention_quantization.inspection.core.storage import Storage


class StorageTests(unittest.TestCase):
    def test_cpu_copy_is_detached_and_independent(self):
        """CPU snapshots survive subsequent mutation of the source tensor."""
        source = torch.tensor([1., 2.], requires_grad=True)
        saved = Storage("cpu").save(source)
        self.assertFalse(saved.requires_grad)
        self.assertEqual(saved.device.type, "cpu")
        with torch.no_grad():
            source.add_(10)
        torch.testing.assert_close(saved, torch.tensor([1., 2.]))

    def test_live_storage_preserves_identity_and_graph(self):
        """Live storage keeps the original tensor and its gradient connection."""
        source = torch.tensor([2.], requires_grad=True)
        saved = Storage("live").save(source.square())
        self.assertIs(Storage("live").save(saved), saved)
        torch.testing.assert_close(torch.autograd.grad(saved.sum(), source)[0], torch.tensor([4.]))

    def test_disk_files_are_unique_and_round_trip(self):
        """Disk storage preserves existing files, tensor values, shapes, and dtype."""
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "existing.txt"
            sentinel.write_text("keep")
            source = torch.arange(6, dtype=torch.float64).reshape(2, 3)
            storage = Storage("disk", directory)
            first, second = storage.save(source), storage.save(source)
            self.assertNotEqual(first, second)
            torch.testing.assert_close(torch.load(first, weights_only=True), source)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_invalid_policies_fail_before_storage(self):
        """Reject unsupported modes and disk storage without a destination."""
        for mode, directory in [("unknown", None), ("disk", None)]:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                Storage(mode, directory)
