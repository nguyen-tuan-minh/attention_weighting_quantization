"""Results retain forward and module-call identities, including cached generation."""
from dataclasses import dataclass, field


@dataclass
class ForwardCapture:
    """One forward's accumulated captures, identified by its session index.

    tensors: (layer, target) -> module-call list of Tensor, disk Path, or None
    for attention reduced immediately. measurements: (layer, measurement name)
    -> call/objective list of per-sample results. gradients: (layer, target)
    -> tensors/paths appended in objective order then module-call order.
    Storage determines device and graph ownership of retained tensors.
    """
    index: int
    tensors: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    gradients: dict = field(default_factory=dict)
