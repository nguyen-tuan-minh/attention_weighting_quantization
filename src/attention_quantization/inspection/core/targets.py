"""Model-specific descriptions of observable tensors."""
from dataclasses import dataclass
from typing import Callable, Any

from torch import Tensor, nn


@dataclass(frozen=True)
class CaptureTarget:
    """A module boundary and extractor; extractors must return a live tensor."""

    name: str
    layer: int
    module: nn.Module
    extract: Callable[[Any], Tensor]
    before: bool = False
