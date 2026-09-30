"""Model loading helpers that are independent of a particular analysis task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration


_MODEL_CLASSES = {
    "AutoModelForVision2Seq": AutoModelForVision2Seq,
    "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
}


def load_huggingface_model(
    model_path: str | Path,
    *,
    model_class: str = "AutoModelForVision2Seq",
    torch_dtype: str | torch.dtype = "float16",
    device_map: str | dict[str, Any] = "auto",
    attn_implementation: str | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Load a configured Transformers vision-language model and its processor.

    ``model_class`` can be changed in model.yaml when a checkpoint is not
    supported by the generic vision-to-sequence auto class.
    """
    model_class_type = _MODEL_CLASSES.get(model_class)
    if model_class_type is None:
        raise ValueError(f"Unsupported model_class {model_class!r}; choose from {tuple(_MODEL_CLASSES)}")

    if isinstance(torch_dtype, str):
        resolved_dtype = getattr(torch, torch_dtype, None)
    else:
        resolved_dtype = torch_dtype
    if resolved_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("torch_dtype must be float16, bfloat16, or float32")

    path = str(Path(model_path).expanduser())
    processor = AutoProcessor.from_pretrained(path)
    load_kwargs: dict[str, Any] = {"torch_dtype": resolved_dtype, "device_map": device_map}
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = model_class_type.from_pretrained(path, **load_kwargs)
    model.eval()
    return model, processor
