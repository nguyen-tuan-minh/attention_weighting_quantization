import torch


class GradientCollector:
    """Differentiate an explicit scalar objective without updating parameter .grad."""

    def compute(self, loss, tensors, *, retain_graph=False):
        """Return d(loss)/d(tensor) in input order, preserving tensor shapes.

        Requires a scalar objective and nonempty live grad-enabled captures, all
        participating in its graph. retain_graph supports subsequent objectives.
        Raises on disconnected captures; does not accumulate parameter gradients.
        """
        if loss.numel() != 1:
            raise ValueError("Gradient objective must be scalar")
        if not tensors or any(not tensor.requires_grad for tensor in tensors):
            raise RuntimeError("Gradient capture requires live, grad-enabled tensors")
        return torch.autograd.grad(loss, tensors, retain_graph=retain_graph, allow_unused=False)
