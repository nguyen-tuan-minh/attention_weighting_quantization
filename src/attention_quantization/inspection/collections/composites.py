"""Public collector definitions and coordination of shared model captures."""
from copy import deepcopy

from ..core.composite import CompositeCollector
from ..core.results import CollectionCapture
from .attention import iga, attention_mass
from .measurements import importance, overlap


def _mask(value):
    """Own a detached boolean mask so caller mutations cannot alter configuration."""
    import torch
    return torch.as_tensor(value, dtype=torch.bool).detach().clone()


class Collection:
    """Configuration and calculations for one public collector definition.

    Subclasses declare semantic tensor dependencies and return measurements.
    Configuration is copied per forward; all accumulated state belongs to that
    forward's CollectionCapture. No module hooks or model tensors live here.
    """

    needs_grad = False
    default_label = None

    def __init__(self, *, name=None, label=None):
        """Choose the collector namespace and optional declared measurement label.

        Subclasses set default_label once. label overrides that
        declaration; raw collectors have no measurement output by default.
        """
        self.name = type(self).__name__ if name is None else name
        self.label = self.default_label if label is None else label
        Collection.validate(self)

    @property
    def labels(self):
        """List declared measurement labels, including component labels in composites."""
        return () if self.label is None else (self.label,)

    def measurement_key(self, layer):
        """Return this collector's declared output key for a decoder layer."""
        if self.label is None:
            raise ValueError("This collector has no declared measurement output")
        return layer, self.label

    def _validate_names(self):
        """Validate namespace and single-output declaration without inspecting children."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Collector name must be a nonempty string")
        if (self.label is not None or self.default_label is not None) and (
                not isinstance(self.label, str) or not self.label.strip()):
            raise ValueError("label must be a nonempty string")

    def required_targets(self):
        """Return semantic tensor names resolved by the model inspector."""
        raise NotImplementedError

    def validate(self):
        """Validate analysis configuration before hooks attach or a forward begins."""
        self._validate_names()

    def snapshot(self):
        """Copy configuration for one forward, isolating later collector changes."""
        result = deepcopy(self)
        result.validate()
        return result

    def wants_gradient(self, target):
        """Return whether this collector requires derivatives for a tensor target."""
        return False

    def measure_tensor(self, target, tensor):
        """Return (measurement name, values), or None to retain the raw tensor."""
        return None

    def process_gradient(self, result, key, call, gradient, store, load):
        """Consume a derivative when needed; raw collectors retain no extra statistics."""


class TensorCapture(Collection):
    """Request arbitrary exposed tensor targets, optionally with raw derivatives."""

    def __init__(self, targets, *, gradients=False, name=None):
        """Store semantic names and declare whether their live graphs are needed."""
        super().__init__(name=name)
        self.targets = (targets,) if isinstance(targets, str) else tuple(targets)
        self.needs_grad = gradients
        self.validate()

    def validate(self):
        """Reject empty or non-string semantic tensor requests."""
        super().validate()
        if not self.targets or any(not isinstance(name, str) or not name for name in self.targets):
            raise ValueError("TensorCapture requires nonempty semantic target names")

    def required_targets(self):
        """Return the requested tensor names; the group deduplicates shared dependencies."""
        return self.targets

    def wants_gradient(self, target):
        """Request derivatives only when configured and the tensor is a dependency."""
        return self.needs_grad and target in self.targets


class AttentionWeightsCollector(TensorCapture):
    """Capture full post-softmax attention matrices."""

    def __init__(self, *, gradients=False, name=None):
        """Select attention weights with optional gradient retention."""
        super().__init__("attention.weights", gradients=gradients, name=name)


class ValueCollector(TensorCapture):
    """Capture raw attention value-projection outputs."""

    def __init__(self, *, gradients=False, name=None):
        """Select value projections with optional gradient retention."""
        super().__init__("attention.value", gradients=gradients, name=name)


class IGACollector(Collection):
    """Own query/image masks and reduce attention to image-token scores."""

    default_label = "iga"

    def __init__(self, *, query_mask, image_mask, name=None, label=None):
        """Own [sequence] or [batch, sequence] masks for attention query/key selection."""
        super().__init__(name=name, label=label)
        self.query_mask, self.image_mask = _mask(query_mask), _mask(image_mask)

    def required_targets(self):
        """Request post-softmax attention weights."""
        return ["attention.weights"]

    def measure_tensor(self, target, tensor):
        """Average selected queries and heads into one image-token vector per sample."""
        return self.label, iga(tensor, self.query_mask, self.image_mask)


class AttentionMassCollector(IGACollector):
    """Own modality masks and calculate image/text attention totals."""

    default_label = "attention_mass"

    def __init__(self, *, query_mask, image_mask, text_mask, name=None, label=None):
        """Configure answer queries and disjoint modality key masks."""
        super().__init__(query_mask=query_mask, image_mask=image_mask, name=name, label=label)
        self.text_mask = _mask(text_mask)

    def measure_tensor(self, target, tensor):
        """Sum selected image/text weights and count contributing head/query rows."""
        return self.label, attention_mass(tensor, self.query_mask, self.image_mask, self.text_mask)


class GradientImportanceCollector(Collection):
    """Own the activation target, masks, reduction, and objective accumulation."""

    needs_grad = True
    default_label = "gradient_importance"

    def __init__(self, *, gradient_target="attention.value", image_mask=None,
                 reduction="l2", accumulate=False, name=None, label=None):
        """Configure token scores; sum_squared supports per-token objective accumulation."""
        super().__init__(name=name, label=label)
        self.gradient_target = gradient_target
        self.image_mask = _mask(image_mask) if image_mask is not None else None
        self.reduction, self.accumulate = reduction, accumulate
        self.validate()

    def validate(self):
        """Validate reduction and target before differentiating any objective."""
        super().validate()
        if self.reduction not in {"l2", "sum_squared"}:
            raise ValueError("reduction must be l2 or sum_squared")
        if not isinstance(self.gradient_target, str) or not self.gradient_target:
            raise ValueError("gradient_target must be a semantic tensor name")

    def required_targets(self):
        """Request the activation against which objectives are differentiated."""
        return [self.gradient_target]

    def wants_gradient(self, target):
        """Retain only the configured activation graph for scoring."""
        return target == self.gradient_target

    def process_gradient(self, result, key, call, gradient, store, load):
        """Reduce and accumulate derivatives within this collector's forward result.

        Accumulators map (target key, call) to detached per-sample scores. Store
        and load callbacks keep device/disk policy outside this collector.
        """
        options = {} if self.image_mask is None else {"image_mask": self.image_mask}
        scores = importance(gradient, options, self.reduction)
        measurement_key = self.measurement_key(key[0])
        if self.accumulate:
            accumulator_key = (key, call)
            previous = result.accumulators.get(accumulator_key)
            if previous is not None:
                scores = [a + b for a, b in zip(previous, scores)]
            result.accumulators[accumulator_key] = [score.detach() for score in scores]
            values = result.measurements.setdefault(measurement_key, [])
            while len(values) <= call:
                values.append(None)
            values[call] = store(scores)
        else:
            result.measurements.setdefault(measurement_key, []).append(store(scores))
        return scores


class OverlapCollector(Collection):
    """Compose IGA and gradient importance using collector-owned configuration."""

    needs_grad = True
    default_label = "overlap"

    def __init__(self, *, query_mask, image_mask, gradient_target="attention.value",
                 top_fraction=0.1, reduction="l2", accumulate=False, name=None,
                 label=None, iga_label=None, importance_label=None):
        """Configure component calculations and independently named measurement outputs."""
        super().__init__(name=name, label=label)
        self.iga = IGACollector(query_mask=query_mask, image_mask=image_mask, label=iga_label)
        self.importance = GradientImportanceCollector(gradient_target=gradient_target,
            image_mask=image_mask, reduction=reduction, accumulate=accumulate,
            label=importance_label)
        self.top_fraction = top_fraction
        self.validate()

    def validate(self):
        """Validate component calculations and the top-token fraction."""
        super().validate()
        self.iga.validate()
        self.importance.validate()
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Composite measurement labels must be unique")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")

    @property
    def labels(self):
        """Expose all component and aggregate outputs in the composite namespace."""
        return (*self.iga.labels, *self.importance.labels, self.label)

    def required_targets(self):
        """Combine attention and activation dependencies before group deduplication."""
        return self.iga.required_targets() + self.importance.required_targets()

    def wants_gradient(self, target):
        """Delegate derivative selection to gradient importance."""
        return self.importance.wants_gradient(target)

    def measure_tensor(self, target, tensor):
        """Reduce attention and preserve the activation used for derivatives."""
        if target.name in self.iga.required_targets():
            return self.iga.measure_tensor(target, tensor)
        return None

    def process_gradient(self, result, key, call, gradient, store, load):
        """Compute importance then compare it with this collector's matching IGA."""
        scores = self.importance.process_gradient(result, key, call, gradient, store, load)
        attention = load(result.measurements[self.iga.measurement_key(key[0])][call])
        values = overlap(attention, scores, self.top_fraction)
        result.measurements.setdefault(self.measurement_key(key[0]), []).append(values)


class CollectionGroup(CompositeCollector):
    """Coordinate public definitions over a single deduplicated tensor stream."""

    def __init__(self, definitions):
        """Instantiate classes and validate unique result names before model binding."""
        super().__init__([])
        self.definitions = []
        for definition in definitions:
            if isinstance(definition, type) and issubclass(definition, Collection):
                try:
                    definition = definition()
                except TypeError as error:
                    raise TypeError("Pass a configured collector instance when constructor arguments are required") from error
            if not isinstance(definition, Collection):
                raise TypeError("targets must contain collector classes or instances")
            definition.validate()
            self.definitions.append(definition)
        if not self.definitions:
            raise ValueError("Specify at least one collector target")
        names = [definition.name for definition in self.definitions]
        if len(names) != len(set(names)):
            raise ValueError("Collector names must be unique; assign name= for multiple instances")
        # Dependencies are fixed when hooks bind; mutable configuration may only
        # change masks/calculations, never the names or required model tensors.
        self.dependencies = {d.name: tuple(d.required_targets()) for d in self.definitions}
        self.gradient_dependencies = {
            d.name: tuple(target for target in d.required_targets() if d.wants_gradient(target))
            for d in self.definitions
        }
        self.needs_grad = any(d.needs_grad for d in self.definitions)

    def required_targets(self):
        """Return the union of semantic tensor dependencies."""
        return set(name for targets in self.dependencies.values() for name in targets)

    def bind(self, inspector, targets):
        """Attach one model-specific primitive to the deduplicated module targets."""
        self.children = [inspector.tensor_collector(targets)]

    def begin_forward(self, record):
        """Snapshot each collector's configuration into a distinct forward namespace."""
        snapshots = [definition.snapshot() for definition in self.definitions]
        for collector in snapshots:
            if (collector.name not in self.dependencies
                    or tuple(collector.required_targets()) != self.dependencies[collector.name]
                    or tuple(target for target in collector.required_targets() if collector.wants_gradient(target))
                    != self.gradient_dependencies[collector.name]):
                raise ValueError("Collector names and dependencies cannot change after session construction")
            record.collections[collector.name] = CollectionCapture(collector)

    def wants_gradient(self, record, target):
        """Determine whether any snapshot needs a derivative of this target."""
        return any(result.collector.wants_gradient(target) for result in record.collections.values())

    def process_tensor(self, record, target, tensor, call, store):
        """Fan out a shared tensor; return raw-retention intent and stored measurements."""
        keep_raw, measured = False, {}
        for name, result in record.collections.items():
            collector = result.collector
            if target.name not in collector.required_targets():
                continue
            value = collector.measure_tensor(target, tensor)
            if value is None:
                keep_raw = True
            else:
                measurement, values = value
                stored = store(values)
                result.measurements.setdefault((target.layer, measurement), []).append(stored)
                measured[name] = {measurement: stored}
        return keep_raw, measured

    def process_gradient(self, record, key, call, gradient, store, load):
        """Dispatch one shared derivative to each interested collector snapshot."""
        for result in record.collections.values():
            if result.collector.wants_gradient(key[1]):
                result.collector.process_gradient(result, key, call, gradient, store, load)
