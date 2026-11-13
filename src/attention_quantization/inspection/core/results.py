"""Forward identities, shared tensor captures, and isolated collector results."""
from dataclasses import dataclass, field


@dataclass
class CollectionCapture:
    """One collector's state for a forward, owned by the enclosing result.

    collector is a configuration snapshot. measurements maps (layer, measurement)
    to call/objective lists of per-sample values. accumulators holds detached
    per-call statistics for objectives that request accumulation.
    """

    collector: object = field(repr=False)
    measurements: dict = field(default_factory=dict)
    accumulators: dict = field(default_factory=dict, repr=False)


@dataclass
class ForwardCapture:
    """One forward's shared captures and named collector results.

    tensors maps (layer, semantic target) to module-call lists of tensor/path or
    None for reduced-only captures. gradients uses the same keys with values
    appended in objective then call order. collections maps unique collector
    names to CollectionCapture. Storage controls tensor device and graph ownership.
    """

    index: int
    tensors: dict = field(default_factory=dict)
    gradients: dict = field(default_factory=dict)
    collections: dict = field(default_factory=dict)
