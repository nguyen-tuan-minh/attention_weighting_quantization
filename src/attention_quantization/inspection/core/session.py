"""Capture scope coordinating hooks, graph ownership, and derived measurements."""
import math
import torch
from .results import ForwardCapture
from .storage import Storage
from ..collections.attention import iga, attention_mass, mask_for


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
        supported = {None, "iga", "attention_mass", "gradient_importance", "overlap"}
        if collection not in supported:
            raise ValueError(f"Unknown collection: {collection}")
        self.inspector, self.collection, self.options = inspector, collection, options
        self.needs_grad = gradients or collection in {"gradient_importance", "overlap"}
        self.storage = Storage(storage or ("live" if self.needs_grad else "cpu"), directory)
        requested = list(targets or [])
        if collection in {"iga", "attention_mass", "overlap"}:
            requested.append("attention.weights")
            for key in ("query_mask", "image_mask"):
                if key not in options:
                    raise ValueError(f"{collection} requires {key}")
        if collection == "attention_mass" and "text_mask" not in options:
            raise ValueError("attention_mass requires text_mask")
        self.gradient_target = options.get("gradient_target", "attention.value")
        if collection == "overlap" and not 0 < options.get("top_fraction", 0.1) <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        if collection in {"gradient_importance", "overlap"}:
            requested.append(self.gradient_target)
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
        self.collector = inspector.collection_collector(selected, collection)
        # results: list[ForwardCapture], accumulated once per enabled forward;
        # handles: model-level hooks removed on scope exit or setup failure.
        self.results = []
        self.handles = []
        # (forward index, (layer, target), module-call index) -> live Tensor.
        # Private graph references are released after final differentiation,
        # clear(), or close(); live public results have independent ownership.
        self._live = {}
        self._entered = False
        self.enabled = True

    def __enter__(self):
        """Attach hooks and claim the inspector; roll back if setup fails."""
        if self._entered or self.inspector._active:
            raise RuntimeError("Only one capture session may be active per inspector")
        self.inspector._active = self._entered = True
        try:
            def begin(module, inputs):
                """Create an enabled forward's result without changing its inputs."""
                if self.enabled:
                    self.results.append(ForwardCapture(len(self.results)))
            self.handles.append(self.inspector.model.register_forward_pre_hook(begin))
            self.collector.start(self._publish)
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
        if self.needs_grad and (self.collection not in {"gradient_importance", "overlap"}
                                or target.name == self.gradient_target):
            self._live[(record.index, key, call)] = tensor
        # Composite attention measurements are reduced immediately. Explicit raw
        # capture still retains the weights when no collection was requested.
        if target.name == "attention.weights" and self.collection in {"iga", "attention_mass", "overlap"}:
            if self.collection == "attention_mass":
                value = attention_mass(tensor, self.options["query_mask"],
                                       self.options["image_mask"], self.options["text_mask"])
                name = "attention_mass"
            else:
                value = iga(tensor, self.options["query_mask"], self.options["image_mask"])
                name = "iga"
            record.measurements.setdefault((target.layer, name), []).append(self._store_tree(value))
            calls.append(None)
        else:
            calls.append(self.storage.save(tensor))

    def compute_gradients(self, loss, *, forward=-1, retain_graph=False):
        """Differentiate a scalar objective against one forward's live captures.

        Args:
            loss: Grad-enabled scalar depending on every captured target.
            forward: Index into results; defaults to the most recent forward.
            retain_graph: Preserve the graph for another objective when True.

        Returns:
            The forward's gradient dictionary, keyed by (layer, target), with
            stored tensors appended in objective order then module-call order.

        Updates gradients and derived measurements without modifying parameter
        .grad fields. Releases private graph references unless retaining them.
        Must run inside an entered, gradient-enabled session.
        """
        if not self._entered or not self.needs_grad:
            raise RuntimeError("Compute gradients inside a gradient-enabled capture scope")
        record = self.results[forward]
        identities = [identity for identity in self._live if identity[0] == record.index]
        values = [self._live[identity] for identity in identities]
        gradients = self.inspector.gradient_collector().compute(loss, values, retain_graph=retain_graph)
        for (_, key, call), gradient in zip(identities, gradients):
            record.gradients.setdefault(key, []).append(self.storage.save(gradient))
            if self.collection in {"gradient_importance", "overlap"}:
                squared = gradient.float().square().sum(-1)
                if "image_mask" in self.options:
                    mask = mask_for(self.options["image_mask"], squared, -1)
                    scores = [sample[selected].sqrt() for sample, selected in zip(squared, mask)]
                else:
                    scores = list(squared.sqrt())
                record.measurements.setdefault((key[0], "gradient_importance"), []).append(self._store_tree(scores))
                if self.collection == "overlap":
                    # Compute on live scores before optional disk storage.
                    attention = record.measurements[(key[0], "iga")][call]
                    if self.storage.mode == "disk":
                        attention = [torch.load(path, weights_only=True) for path in attention]
                    fraction = self.options.get("top_fraction", 0.1)
                    if not 0 < fraction <= 1:
                        raise ValueError("top_fraction must be in (0, 1]")
                    overlaps = []
                    for a, b in zip(attention, scores):
                        count = max(1, math.ceil(a.numel() * fraction))
                        left = set(a.topk(count).indices.tolist())
                        right = set(b.topk(count).indices.tolist())
                        overlaps.append(len(left & right) / count)
                    record.measurements.setdefault((key[0], "overlap"), []).append(overlaps)
        if not retain_graph:
            for identity in identities:
                del self._live[identity]
        return record.gradients

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
