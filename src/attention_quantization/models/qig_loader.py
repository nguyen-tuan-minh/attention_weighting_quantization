"""Load original and QIG-quantized LLaVA checkpoints."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def resolve_llava_model_name(
    checkpoint_dir: str | Path,
    *,
    configured_name: str | None = None,
) -> str:
    """Resolve the LLaVA family identifier expected by the upstream builder.

    Prefer size/version markers in the checkpoint path or its saved config so
    expert and amateur checkpoints can each resolve independently. The config
    value is a fallback for custom directory names.
    """
    checkpoint = Path(checkpoint_dir).expanduser()
    config_path = checkpoint / "config.json"
    config_name = ""
    if config_path.is_file():
        try:
            config_name = str(json.loads(config_path.read_text(encoding="utf-8")).get("_name_or_path", ""))
        except (OSError, json.JSONDecodeError):
            pass

    candidates = (str(checkpoint), config_name, configured_name or "")
    for candidate in candidates:
        match = re.search(r"llava[-_]v?1[.]5[-_](7b|13b)", candidate, re.IGNORECASE)
        if match:
            return f"llava-v1.5-{match.group(1).lower()}"

    # The LLaVA 1.5 13B checkpoint uses the 5120-wide, 40-layer LLaMA-2
    # decoder; 7B uses 4096-wide, 32-layer. This handles renamed local copies.
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if int(config.get("hidden_size", 0)) == 5120 and int(config.get("num_hidden_layers", 0)) == 40:
                return "llava-v1.5-13b"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if configured_name:
        return configured_name
    return "llava-v1.5-7b"


def load_qig_llava_model(
    checkpoint_dir: str | Path,
    *,
    qig_source_dir: str | Path,
    device: str = "cuda:0",
    attn_implementation: str = "eager",
    configured_model_name: str | None = None,
) -> tuple[Any, Any]:
    """Load original LLaVA 1.5 weights with QIG's LLaVA implementation.

    Returns the LMMS-Eval model wrapper and QIG's multimodal processing model.
    This path supports LLaVA 1.5 7B and 13B checkpoints, whose architecture is
    not the Transformers ``LlavaForConditionalGeneration``.
    """
    from attention_quantization.quantization.qig import load_qig_runtime

    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    has_weights = any(checkpoint.glob("pytorch_model*.bin")) or any(checkpoint.glob("model*.safetensors"))
    if not (checkpoint / "config.json").is_file() or not has_weights:
        raise FileNotFoundError(f"Complete LLaVA checkpoint files were not found under {checkpoint}")
    model_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    architectures = model_config.get("architectures", [])
    supported_model_type = model_config.get("model_type") in {"llava", "llava_llama"}
    supported_architecture = not architectures or any(
        name == "LlavaLlamaForCausalLM" for name in architectures
    )
    if not supported_model_type or not supported_architecture:
        raise ValueError(
            f"{checkpoint} is not a supported LLaVA 1.5 checkpoint format expected by QIG; "
            f"found model_type={model_config.get('model_type')!r}, architectures={architectures!r}"
        )

    runtime = load_qig_runtime(qig_source_dir)
    model_class = runtime.get_model("llava")
    model_name = resolve_llava_model_name(checkpoint, configured_name=configured_model_name)
    model_args = (
        f"pretrained={checkpoint},model_name={model_name},"
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
    configured_model_name: str | None = None,
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
    model_name = resolve_llava_model_name(checkpoint, configured_name=configured_model_name)
    model_args = f"pretrained={checkpoint},model_name={model_name}"
    return model_class.create_from_arg_string(
        model_args,
        {"batch_size": batch_size, "device": device},
    )
