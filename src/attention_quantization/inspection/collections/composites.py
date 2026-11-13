"""High-level collectors own dependencies, validation, and measurement workflows."""
from ..core.composite import CompositeCollector
from .attention import iga, attention_mass
from .measurements import importance, overlap


class Collection(CompositeCollector):
    """Raw capture workflow, also the extension contract for derived collections."""

    needs_grad = False
    context_keys = ()

    def configure(self, options):
        """Configure immutable capture dependencies before binding module hooks."""
        self.gradient_target = options.get("gradient_target", "attention.value")

    def bind(self, inspector, targets):
        """Bind deduplicated targets to one model-specific tensor collector."""
        self.children = [inspector.tensor_collector(targets)]

    def required_targets(self):
        """Return semantic dependencies added to explicit user targets."""
        return []

    def validate(self, options, *, require_context):
        """Validate collection context before setup or a measured forward."""
        if require_context:
            for name in self.context_keys:
                if name not in options:
                    raise ValueError(f"{type(self).__name__} requires {name}")

    def wants_gradient(self, target):
        """Select every raw target when the session explicitly enables gradients."""
        return True

    def measure_tensor(self, target, tensor, options):
        """Return None to preserve a raw capture, or (name, values) for a reduction."""
        return None

    def validate_gradient(self, reduction, accumulate):
        """Reject unknown reductions before differentiating the objective."""
        if reduction not in {"l2", "sum_squared"}:
            raise ValueError("reduction must be l2 or sum_squared")

    def process_gradient(self, record, key, call, gradient, reduction, accumulate, store, load):
        """Leave raw derivatives untouched; derived workflows override this callback."""
        return None


class IGACollector(Collection):
    """Select attention dependencies and compute image-guided attention vectors."""

    context_keys = ("query_mask", "image_mask")

    def required_targets(self):
        """Request post-softmax attention weights."""
        return ["attention.weights"]

    def measure_tensor(self, target, tensor, options):
        """Average selected queries and heads, preserving other requested tensors."""
        if target.name == "attention.weights":
            return "iga", iga(tensor, options["query_mask"], options["image_mask"])
        return None


class AttentionMassCollector(IGACollector):
    """Compute per-modality attention totals from selected answer queries."""

    context_keys = ("query_mask", "image_mask", "text_mask")

    def measure_tensor(self, target, tensor, options):
        """Sum disjoint image/text key masses for each batch item."""
        if target.name == "attention.weights":
            return "attention_mass", attention_mass(tensor, options["query_mask"], options["image_mask"], options["text_mask"])
        return None


class GradientImportanceCollector(Collection):
    """Own activation dependencies, gradient reductions, and objective accumulation."""

    needs_grad = True

    def required_targets(self):
        """Request the activation against which objectives are differentiated."""
        return [self.gradient_target]

    def wants_gradient(self, target):
        """Retain only the configured activation's graph for importance scoring."""
        return target == self.gradient_target

    def process_gradient(self, record, key, call, gradient, reduction, accumulate, store, load):
        """Reduce derivatives and update this forward's per-call importance results.

        Accumulators belong to record: (target key, call, reduction) maps to
        detached per-sample vectors. No state is shared between forwards.
        Storage callbacks keep device/disk policy outside this collection.
        """
        scores = importance(gradient, record.options, reduction)
        measurement_key = (key[0], "gradient_importance")
        if accumulate:
            accumulator_key = (key, call, reduction)
            previous = record.accumulators.get(accumulator_key)
            if previous is not None:
                scores = [a + b for a, b in zip(previous, scores)]
            record.accumulators[accumulator_key] = [score.detach() for score in scores]
            values = record.measurements.setdefault(measurement_key, [])
            while len(values) <= call:
                values.append(None)
            values[call] = store(scores)
        else:
            record.measurements.setdefault(measurement_key, []).append(store(scores))
        return scores


class OverlapCollector(Collection):
    """Compose IGA and gradient importance, then compare selected image tokens."""

    needs_grad = True

    def configure(self, options):
        """Construct component workflows sharing the same resolved capture stream."""
        super().configure(options)
        self.iga = IGACollector([])
        self.importance = GradientImportanceCollector([])
        self.iga.configure(options)
        self.importance.configure(options)

    def required_targets(self):
        """Combine attention and activation dependencies before deduplication."""
        return self.iga.required_targets() + self.importance.required_targets()

    def validate(self, options, *, require_context):
        """Validate component masks and the top-token selection fraction."""
        self.iga.validate(options, require_context=require_context)
        self.importance.validate(options, require_context=require_context)
        if not 0 < options.get("top_fraction", 0.1) <= 1:
            raise ValueError("top_fraction must be in (0, 1]")

    def wants_gradient(self, target):
        """Delegate graph retention to gradient importance."""
        return self.importance.wants_gradient(target)

    def measure_tensor(self, target, tensor, options):
        """Delegate attention reduction to IGA."""
        return self.iga.measure_tensor(target, tensor, options)

    def process_gradient(self, record, key, call, gradient, reduction, accumulate, store, load):
        """Compute importance, load matching IGA, and append per-sample overlap."""
        scores = self.importance.process_gradient(record, key, call, gradient, reduction, accumulate, store, load)
        attention = load(record.measurements[(key[0], "iga")][call])
        values = overlap(attention, scores, record.options.get("top_fraction", 0.1))
        record.measurements.setdefault((key[0], "overlap"), []).append(values)


def compose(inspector, targets, collection):
    """Select a workflow; model inspectors may override this factory for new types."""
    classes = {None: Collection, "iga": IGACollector, "attention_mass": AttentionMassCollector,
               "gradient_importance": GradientImportanceCollector, "overlap": OverlapCollector}
    if collection not in classes:
        raise ValueError(f"Unknown collection: {collection}")
    result = classes[collection]([])
    if targets:
        result.bind(inspector, targets)
    return result
