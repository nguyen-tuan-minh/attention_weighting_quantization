"""LLaVA collection extension points."""
from ...collectors.tensor import TensorCollector
from ...collectors.gradient import GradientCollector


class LlavaTensorCollector(TensorCollector):
    """Capture tensors using LLaVA target extractors."""


class LlavaGradientCollector(GradientCollector):
    """Differentiate LLaVA captures with their native tensor layouts."""
