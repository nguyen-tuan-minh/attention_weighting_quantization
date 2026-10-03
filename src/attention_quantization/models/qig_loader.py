"""Load original and QIG-quantized LLaVA checkpoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def load_qig_llava_model(
    checkpoint_dir: str | Path,
    *,
    qig_source_dir: str | Path,
    device: str = "cuda:0",
    attn_implementation: str = "eager",
) -> tuple[Any, Any]:
    """Load original LLaVA 1.5 weights with QIG's LLaVA implementation.

    Returns the LMMS-Eval model wrapper and QIG's multimodal processing model.
    This path is needed for ``liuhaotian/llava-v1.5-7b`` checkpoints, whose
    architecture is not the Transformers ``LlavaForConditionalGeneration``.
    """
    from attention_quantization.quantization.qig import load_qig_runtime

    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    has_weights = any(checkpoint.glob("pytorch_model*.bin")) or any(checkpoint.glob("model*.safetensors"))
    if not (checkpoint / "config.json").is_file() or not has_weights:
        raise FileNotFoundError(f"Complete LLaVA checkpoint files were not found under {checkpoint}")
    model_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    architectures = model_config.get("architectures", [])
    if model_config.get("model_type") != "llava" or (
        architectures and not any(name == "LlavaLlamaForCausalLM" for name in architectures)
    ):
        raise ValueError(
            f"{checkpoint} is not the original LLaVA 1.5 checkpoint format expected by QIG; "
            f"found model_type={model_config.get('model_type')!r}, architectures={architectures!r}"
        )

    runtime = load_qig_runtime(qig_source_dir)
    model_class = runtime.get_model("llava")
    model_args = (
        f"pretrained={checkpoint},model_name=llava-v1.5-7b,"
        f"attn_implementation={attn_implementation}"
    )
    lm = model_class.create_from_arg_string(
        model_args,
        {"batch_size": 1, "device": device, "device_map": device},
    )
    process_class = runtime.get_process_model("llava")
    process_model = process_class(lm._model, lm._tokenizer, lm._image_processor)
    return lm, process_model


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
            "The model loading dependencies are unavailable. Run scripts/set_up.sh "
            "to install the project environment."
        ) from error

    model_class = get_model("llava")
    # QIG's classic LLaVA adapter rejects a `dtype` argument. Its upstream
    # LLaVA builder defaults to float16, matching the saved checkpoint.
    model_args = f"pretrained={checkpoint}"
    return model_class.create_from_arg_string(
        model_args,
        {"batch_size": batch_size, "device": device},
    )
