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

    def collection_collector(self, targets, collection):
        """Compose resolved dependencies once; subclasses may override wiring."""
        from ..collections.composites import compose
        return compose(self, targets, collection)

    def release_attention_hooks(self, targets, enabled):
        """Register model-specific release hooks or reject unsupported release."""
        raise ValueError("This inspector does not support attention-output release")

    def capture(self, *, targets=None, layers=None, collection=None,
                gradients=False, storage=None, directory=None, **options):
        """Create an unentered capture session without registering hooks.

        Select semantic targets or a named collection and optional layer indices.
        gradients enables differentiation of explicit captures; storage selects
        live, CPU, or disk retention (disk requires directory). Collection options
        include query/image/text masks, gradient_target, and top_fraction.
        The returned context manager owns registration and cleanup.
        """
        return CaptureSession(self, targets, layers, collection, gradients,
                              storage, directory, options)
