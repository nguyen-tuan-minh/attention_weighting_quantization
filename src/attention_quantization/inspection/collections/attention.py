"""Reductions over [batch, heads, queries, keys] attention tensors."""
import torch


def mask_for(mask, tensor, axis):
    """Normalize [sequence] or [batch, sequence] masks on the tensor device.

    Returns bool [batch, tensor.shape[axis]]. Rejects shape mismatches and samples
    with no selected positions; does not modify the caller's mask.
    """
    mask = torch.as_tensor(mask, device=tensor.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(tensor.shape[0], -1)
    if tuple(mask.shape) != (tensor.shape[0], tensor.shape[axis]):
        raise ValueError("Mask must match batch and sequence dimensions")
    if not mask.any(dim=1).all():
        raise ValueError("Each sample must select at least one token")
    return mask


def iga(weights, query_mask, image_mask):
    """Average [batch, heads, queries, keys] attention over heads/selected queries.

    Returns one float32 [image_tokens] tensor per sample on the weights device;
    keeps autograd connections. Mask shapes follow mask_for's contract.
    """
    if weights.ndim != 4:
        raise ValueError("Attention must have shape [batch, heads, queries, keys]")
    queries = mask_for(query_mask, weights, -2)
    keys = mask_for(image_mask, weights, -1)
    return [sample[:, query, :][:, :, key].float().mean((0, 1))
            for sample, query, key in zip(weights, queries, keys)]


def attention_mass(weights, query_mask, image_mask, text_mask):
    """Sum attention over selected queries and disjoint image/text key masks.

    Reads [batch, heads, queries, keys] weights. Returns a per-sample dictionary
    of scalar image/text tensors and integer head/query row count. Tensors stay
    on the weights device with graph connections; masks follow mask_for.
    """
    queries = mask_for(query_mask, weights, -2)
    images = mask_for(image_mask, weights, -1)
    texts = mask_for(text_mask, weights, -1)
    if (images & texts).any():
        raise ValueError("Image and text key masks must be disjoint")
    result = []
    for sample, query, image, text in zip(weights, queries, images, texts):
        selected = sample[:, query, :].float()
        result.append({"image": selected[:, :, image].sum(),
                       "text": selected[:, :, text].sum(),
                       "query_rows": selected.shape[0] * selected.shape[1]})
    return result
