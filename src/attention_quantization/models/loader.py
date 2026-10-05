"""Model loader selected by the project's model configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_model(
    checkpoint_dir: str | Path,
    model_config: dict[str, Any],
    *,
    repository_root: str | Path,
) -> tuple[Any, Any]:
    """Load the configured model and its multimodal input adapter.

    Model-specific loading stays behind this function so calibration and
    investigation scripts can share one interface as the project grows.
    """
    model_family = model_config.get("model_family", "llava-v1.5")
    if model_family != "llava-v1.5":
        raise ValueError(f"Unsupported model_family {model_family!r}")

    from .qig_loader import load_qig_llava_model

    source_dir = model_config.get("implementation_source_dir") or os.environ.get("QIG_SOURCE_DIR")
    if source_dir is None:
        source_dir = Path(repository_root) / "third_party" / "QIG"
    elif not Path(source_dir).is_absolute():
        source_dir = Path(repository_root) / source_dir

    lm, input_adapter = load_qig_llava_model(
        checkpoint_dir,
        qig_source_dir=source_dir,
        device=model_config.get("device_map", "cuda:0"),
        attn_implementation=model_config.get("attn_implementation", "eager"),
    )
    return lm._model, input_adapter
