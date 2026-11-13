"""Stateless numerical measurements consumed by high-level collectors."""
import math
from .attention import mask_for


def importance(gradient, options, reduction):
    """Reduce [batch, sequence, features] gradients into per-sample scores."""
    squared = gradient.float().square().sum(-1)
    if "image_mask" in options:
        mask = mask_for(options["image_mask"], squared, -1)
        scores = [sample[selected] for sample, selected in zip(squared, mask)]
    else:
        scores = list(squared)
    return [score.sqrt() for score in scores] if reduction == "l2" else scores

def overlap(attention, scores, fraction):
    """Return per-sample top-token intersection fractions."""
    result = []
    for left, right in zip(attention, scores):
        count = max(1, math.ceil(left.numel() * fraction))
        a, b = set(left.topk(count).indices.tolist()), set(right.topk(count).indices.tolist())
        result.append(len(a & b) / count)
    return result
