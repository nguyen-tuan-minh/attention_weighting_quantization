"""Load checkpoints produced by ``scripts/quantize_qig.sh``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def load_qig_quantized_model(
    checkpoint_dir: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 1,
    qig_source_dir: str | Path | None = None,
) -> Any:
    """Load a saved QIG LLaVA checkpoint through QIG's LMMS-Eval adapter.

    The returned object is the LMMS-Eval model wrapper; its underlying
    Transformers model is available as ``wrapper._model``.
    """
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    metadata_path = checkpoint / "qig_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"QIG metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_type") != "llava":
        raise ValueError(f"Unsupported QIG model type: {metadata.get('model_type')!r}")

    source_value = qig_source_dir or os.environ.get("QIG_SOURCE_DIR")
    if source_value:
        source_path = Path(source_value).expanduser().resolve()
        if not (source_path / "main_quant.py").is_file():
            raise FileNotFoundError(f"QIG source was not found: {source_path}")
        sys.path.insert(0, str(source_path))

    try:
        from lmms_eval.models import get_model
    except ImportError as error:
        raise ImportError(
            "QIG dependencies are unavailable. Run this loader with the isolated QIG "
            "environment created by scripts/quantize_qig.sh."
        ) from error

    model_class = get_model("llava")
    dtype = metadata.get("model_dtype", "float16")
    model_args = f"pretrained={checkpoint},dtype={dtype}"
    return model_class.create_from_arg_string(
        model_args,
        {"batch_size": batch_size, "device": device},
    )
