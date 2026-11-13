from abc import ABC, abstractmethod
from .session import CaptureSession


class ModelInspector(ABC):
    """Public entry point; subclasses resolve targets and choose collectors."""

    def __init__(self, model):
        """Keep the supplied model and track exclusive capture-scope ownership."""
        self.model = model
        self._active = False

    @abstractmethod
    def targets(self):
        """Return CaptureTarget definitions for this concrete model."""

    def capture_targets(self):
        """Return available semantic target names mapped to decoder indices."""
        available = {}
        for target in self.targets():
            available.setdefault(target.name, []).append(target.layer)
        return available

    @abstractmethod
    def tensor_collector(self, targets):
        """Build a model-specific collector for resolved CaptureTarget objects."""
        pass

    @abstractmethod
    def gradient_collector(self):
        """Build the model-specific objective differentiation implementation."""
        pass

    def collection_group(self, targets):
        """Resolve public collector definitions into a shared capture workflow.

        Model subclasses may override this factory to adapt collector behavior;
        tensor extraction still uses their tensor_collector implementation.
        """
        from ..collections.composites import CollectionGroup
        return CollectionGroup(targets)

    def release_attention_hooks(self, targets, enabled):
        """Register model-specific release hooks or reject unsupported release."""
        raise ValueError("This inspector does not support attention-output release")

    def capture(self, *, targets, layers=None, storage=None, directory=None,
                dtype=None, scoped=False, sink=None, retain_tensors=True,
                release_attention=False):
        """Create a session from collector classes or configured instances.

        targets owns all analysis settings, including masks and gradient needs.
        Remaining arguments configure selected layers, storage, and lifecycle.
        storage is live, cpu, or disk; disk requires directory. dtype applies to
        detached storage. scoped captures only inside session.forward(). The
        optional synchronous sink receives each completed capture; retain_tensors
        controls raw result retention. release_attention drops returned weights
        after all collection consumers have processed them.
        """
        return CaptureSession(
            self, targets=targets, layers=layers, storage=storage,
            directory=directory, dtype=dtype, scoped=scoped, sink=sink,
            retain_tensors=retain_tensors, release_attention=release_attention,
        )
