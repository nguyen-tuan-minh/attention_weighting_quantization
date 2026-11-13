"""Capture scope coordinating hooks, graph ownership, and derived measurements."""
from contextlib import contextmanager
import torch
from .results import ForwardCapture
from .storage import Storage


class CaptureSession:
    """Own captures for one scope; callers own retained results after exit."""

    def __init__(self, inspector, targets, layers, collection, gradients, storage, directory, options):
        """Resolve targets and validate collection options without attaching hooks.

        `targets` are semantic names and `layers` are decoder indices; None
        selects collection dependencies or all available layers, respectively.
        `options` holds query/image/text masks, gradient_target, and top_fraction.
        Storage defaults to live for gradients and CPU otherwise. Unsupported
        targets, collections, masks missing from options, or policies raise
        ValueError before model execution.
        """
        self.inspector, self.options = inspector, options
        self.collector = inspector.collection_collector([], collection)
        self.collector.configure(options)
        self.needs_grad = gradients or self.collector.needs_grad
        self.storage = Storage(storage or ("live" if self.needs_grad else "cpu"), directory, options.get("dtype"))
        self.scoped = options.get("scoped", False)
        self.collector.validate(options, require_context=not self.scoped)
        self.sink = options.get("sink")
        self.release_attention = options.get("release_attention", False)
        self._in_forward = False
        requested = list(targets or []) + self.collector.required_targets()
        requested = set(requested)
        if not requested:
            raise ValueError("Specify targets or a collection")
        available = inspector.targets()
        selected_layers = set(layers) if layers is not None else {t.layer for t in available}
        selected = [t for t in available if t.name in requested and t.layer in selected_layers]
        missing = {(name, layer) for name in requested for layer in selected_layers} - {
            (t.name, t.layer) for t in selected}
        if not selected or missing:
            raise ValueError(f"Unsupported capture targets/layers: {sorted(missing)}")
        self.targets = selected
        self.collector.bind(inspector, selected)
        # results: list[ForwardCapture], accumulated once per enabled forward;
        # handles: model-level hooks removed on scope exit or setup failure.
        self.results = []
        self.handles = []
        # (forward index, (layer, target), module-call index) -> live Tensor.
        # Private graph references are released after final differentiation,
        # clear(), or close(); live public results have independent ownership.
        self._live = {}
        self._entered = False
        self.enabled = not self.scoped

    def __enter__(self):
        """Attach hooks and claim the inspector; roll back if setup fails."""
        if self._entered or self.inspector._active:
            raise RuntimeError("Only one capture session may be active per inspector")
        self.inspector._active = self._entered = True
        try:
            def begin(module, inputs):
                """Create an enabled forward's result without changing its inputs."""
                if self.enabled:
                    record = ForwardCapture(len(self.results))
                    record.options = dict(self.options)
                    self.results.append(record)
            self.handles.append(self.inspector.model.register_forward_pre_hook(begin))
            def publish(target, tensor):
                """Forward captures to the session while exposing the current capture gate."""
                return self._publish(target, tensor)
            publish.enabled = lambda: self.enabled
            self.collector.start(publish)
            if self.release_attention:
                self.handles.extend(self.inspector.release_attention_hooks(self.targets, lambda: self.enabled))
        except BaseException:
            self.close()
            raise
        return self

    def _store_tree(self, value):
        """Apply storage to tensors in nested lists/dicts, preserving scalars."""
        if torch.is_tensor(value):
            return self.storage.save(value)
        if isinstance(value, list):
            return [self._store_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: self._store_tree(item) for key, item in value.items()}
        return value

    def _load_tree(self, value):
        """Load stored tensor paths recursively for dependent collection calculations."""
        from pathlib import Path
        if isinstance(value, Path):
            return torch.load(value, weights_only=True)
        if isinstance(value, list):
            return [self._load_tree(item) for item in value]
        return value

    def _publish(self, target, tensor):
        """Append one module call to the current forward without changing output.

        Reads a live target tensor. Optionally retains its graph for gradients;
        attention collections reduce [batch, heads, queries, keys] immediately.
        Updates result dictionaries using the configured storage policy.
        """
        if not self.enabled:
            return
        record = self.results[-1]
        key = (target.layer, target.name)
        calls = record.tensors.setdefault(key, [])
        call = len(calls)
        if self.needs_grad and self.collector.wants_gradient(target.name):
            self._live[(record.index, key, call)] = tensor
        measured = self.collector.measure_tensor(target, tensor, record.options)
        if measured is not None:
            name, value = measured
            stored = self._store_tree(value)
            record.measurements.setdefault((target.layer, name), []).append(stored)
            calls.append(None)
        else:
            stored = self.storage.save(tensor)
            calls.append(stored if self.options.get("retain_tensors", True) else None)
        if self.sink is not None:
            self.sink(record.index, target.layer, target.name, call, stored)

    def compute_gradients(self, loss, *, forward=-1, retain_graph=False, reduction="l2", accumulate=False):
        """Differentiate a scalar objective against one forward's live captures.

        Args:
            loss: Grad-enabled scalar depending on every captured target.
            forward: Index into results; defaults to the most recent forward.
            retain_graph: Preserve the graph for another objective when True.
            reduction: l2 for a feature-vector norm or sum_squared for squared
                gradients summed across features in importance collections.
            accumulate: Sum importance scores over objectives per module call.

        Returns:
            The forward's gradient dictionary, keyed by (layer, target), with
            stored tensors appended in objective order then module-call order.

        Updates gradients and derived measurements without modifying parameter
        .grad fields. Releases private graph references unless retaining them.
        Must run inside an entered, gradient-enabled session.
        """
        if not self._entered or not self.needs_grad:
            raise RuntimeError("Compute gradients inside a gradient-enabled capture scope")
        self.collector.validate_gradient(reduction, accumulate)
        record = self.results[forward]
        identities = [identity for identity in self._live if identity[0] == record.index]
        values = [self._live[identity] for identity in identities]
        gradients = self.inspector.gradient_collector().compute(loss, values, retain_graph=retain_graph)
        for (_, key, call), gradient in zip(identities, gradients):
            record.gradients.setdefault(key, []).append(self.storage.save(gradient))
            self.collector.process_gradient(
                record, key, call, gradient, reduction, accumulate,
                self._store_tree, self._load_tree,
            )
        if not retain_graph:
            for identity in identities:
                del self._live[identity]
        return record.gradients

    @contextmanager
    def forward(self, **options):
        """Capture exactly one measured model forward with sample-specific options.

        Requires scoped=True and an entered session. Masks override session
        defaults for this scope. Unmeasured model calls outside the scope are
        ignored. On exit private graph references for the scope are released;
        public live results remain caller-owned. Nested scopes are rejected.
        """
        if not self._entered or not self.scoped or self._in_forward:
            raise RuntimeError("forward requires an entered scoped session and cannot nest")
        previous = self.options
        self.options = dict(previous, **options)
        try:
            self.collector.validate(self.options, require_context=True)
        except BaseException:
            self.options = previous
            raise
        start = len(self.results)
        self._in_forward = self.enabled = True
        try:
            yield self
            if len(self.results) != start + 1:
                raise RuntimeError("A forward scope must execute exactly one model forward")
            record = self.results[-1]
            missing = {(t.layer, t.name) for t in self.targets} - set(record.tensors)
            if missing:
                raise RuntimeError(f"Forward did not capture required targets: {sorted(missing)}")
        finally:
            self.enabled = self._in_forward = False
            self.options = previous
            for identity in list(self._live):
                if identity[0] >= start:
                    del self._live[identity]

    def clear(self):
        """Release results and private graph references; leave hooks attached."""
        self.results.clear()
        self._live.clear()

    def close(self):
        """Remove hooks and private graph references while preserving results."""
        self.collector.close()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._live.clear()
        self.inspector._active = self._entered = False

    def __exit__(self, *exc):
        """Close the scope on success or error without suppressing exceptions."""
        self.close()
