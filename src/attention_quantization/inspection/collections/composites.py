"""Composable model-aware capture dependencies for aggregate measurements.

Tensor targets are resolved once by the inspector. Composite children partition
those targets, so a shared tensor is captured only once.
"""
from ..core.composite import CompositeCollector


class IGACollector(CompositeCollector):
    """Capture dependencies for image-guided attention measurements."""


class AttentionMassCollector(CompositeCollector):
    """Capture dependencies for modality attention totals."""


class GradientImportanceCollector(CompositeCollector):
    """Capture dependencies used by objective-based gradient measurements."""


class OverlapCollector(CompositeCollector):
    """Aggregate attention and gradient-importance capture dependencies."""


def compose(inspector, targets, collection):
    """Build a collector tree from already resolved, deduplicated model targets.

    Partitions attention weights and other targets into distinct children so
    dependencies register once. Uses the inspector's model-specific factories;
    returns an unstarted primitive or composite collector.
    """
    if collection is None:
        return inspector.tensor_collector(targets)
    attention = [target for target in targets if target.name == "attention.weights"]
    other = [target for target in targets if target.name != "attention.weights"]
    children = []
    if attention:
        cls = AttentionMassCollector if collection == "attention_mass" else IGACollector
        children.append(cls([inspector.tensor_collector(attention)]))
    if other:
        children.append(GradientImportanceCollector([inspector.tensor_collector(other)]))
    return OverlapCollector(children) if collection == "overlap" else CompositeCollector(children)
