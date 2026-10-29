"""Compare expert-only, amateur-only, and contrastive multimodal decoding."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import textwrap
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import load_sharegpt4v_dataset  # noqa: E402
from investigation_logging import configure_logging, log  # noqa: E402


_MEMORY_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_TIMINGS: dict[str, list[float]] = {}


def parse_memory_size(value: str) -> int:
    """Convert a memory limit to bytes; an unsuffixed number means decimal GB."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*((?:[kmgt]i?b)|b)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError(
            "use a number with B, KB, MB, GB, TB, KiB, MiB, GiB, or TiB; a bare number means GB"
        )
    unit = (match.group(2) or "GB").lower()
    try:
        return int(float(match.group(1)) * _MEMORY_UNITS[unit])
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(f"invalid memory size: {value!r}") from error


def record_timing(stage: str, elapsed: float) -> None:
    """Collect stage durations for the extensive end-of-run summary."""
    _TIMINGS.setdefault(stage, []).append(elapsed)


def report_timing(stage: str, elapsed: float, *, enabled: bool) -> None:
    """Print one live operation duration when timing was requested."""
    if enabled:
        log(f"[timing] {stage}: {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset.yaml")
    parser.add_argument("--expert-model", type=Path, default=None, help="Expert checkpoint; defaults to model.yaml.")
    parser.add_argument("--amateur-model", type=Path, required=True, help="Amateur checkpoint used for contrastive scores.")
    parser.add_argument("--samples", type=int, default=None, help="Override dataset.yaml calibration_samples.")
    parser.add_argument("--seed", type=int, default=None, help="Override dataset.yaml calibration_seed.")
    parser.add_argument(
        "--decoding-modes", nargs="+", choices=("expert", "amateur", "contrastive"),
        default=("expert", "contrastive"), help="Decoding runs to compare (default: expert contrastive).",
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.5, help="Amateur log-probability subtraction weight.")
    parser.add_argument(
        "--plausibility-threshold", type=float, default=0.1,
        help="Keep tokens whose expert probability is at least this fraction of the expert's top-token probability.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens per answer.")
    parser.add_argument(
        "--preview-inputs", action=argparse.BooleanOptionalAction, default=True,
        help="Save each sample image and print the expert/amateur prompts before decoding (default: on).",
    )
    parser.add_argument("--device", default=None, help="CUDA device used for automatic placement, such as cuda:0; defaults to model.yaml.")
    for role in ("expert", "amateur"):
        for tier in ("gpu", "cpu"):
            parser.add_argument(
                f"--max-{role}-model-memory-{tier}", type=parse_memory_size, default=None,
                help=f"Optional {role} model {tier.upper()} weight-placement limit; bare numbers mean GB.",
            )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/investigations/contrastive-decoding",
        help="Directory for JSONL generations and run metadata.",
    )
    parser.add_argument("--log-level", choices=("none", "normal", "extensive"), default="normal")
    parser.add_argument("--timing", action="store_true", help="Show timing at normal verbosity.")
    parser.add_argument("--quiet-warnings", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace this script's results.jsonl and run_config.json.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Resolve a CLI/config checkpoint path relative to the repository when needed."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else repository_path(path)


def load_runtime(qig_root: Path) -> tuple[Any, Any]:
    """Load LLaVA's builder and multimodal processor from the configured source tree."""
    llava_root = qig_root / "3rdparty/LLaVA-NeXT"
    for path in (qig_root, llava_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from llava.model.builder import load_pretrained_model
    from qmllm.models import get_process_model

    return load_pretrained_model, get_process_model


def make_max_memory(args: argparse.Namespace, role: str, device: str) -> dict[Any, int] | None:
    """Build a complete placement map, preserving automatic budgets for unset tiers."""
    gpu_limit = getattr(args, f"max_{role}_model_memory_gpu")
    cpu_limit = getattr(args, f"max_{role}_model_memory_cpu")
    if gpu_limit is None and cpu_limit is None:
        return None
    if gpu_limit is not None and device == "cpu":
        raise ValueError(f"--max-{role}-model-memory-gpu requires a CUDA --device")

    # Accelerate treats a supplied max_memory mapping as the complete set of
    # placement budgets. Leaving "cpu" out when the user only caps the GPU can
    # make overflow go straight to disk. Start with detected free capacities,
    # then override only the tiers the user explicitly capped.
    from accelerate.utils import get_max_memory

    available = get_max_memory()
    result: dict[Any, int] = {"cpu": cpu_limit if cpu_limit is not None else available["cpu"]}
    if device != "cpu":
        gpu_index = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
        if gpu_index not in available:
            raise ValueError(f"CUDA device {gpu_index} is unavailable for {role} model placement")
        result[gpu_index] = gpu_limit if gpu_limit is not None else available[gpu_index]
    return result


def load_model(
    builder: Any,
    get_process_model: Any,
    checkpoint: Path,
    model_config: dict[str, Any],
    args: argparse.Namespace,
    role: str,
    device: str,
    output_dir: Path,
) -> tuple[Any, Any, Any]:
    """Load one checkpoint and its input adapter, using role-specific placement caps."""
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Model config was not found: {checkpoint / 'config.json'}")
    max_memory = make_max_memory(args, role, device)
    offload_dir = output_dir / ".model-offload" / role
    offload_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log(
        f"Loading {role} model with max_memory={max_memory if max_memory is not None else 'automatic'}",
        level="extensive",
    )
    tokenizer, model, image_processor, _ = builder(
        str(checkpoint), None, "llava-v1.5-7b", device_map="cpu" if device == "cpu" else "auto",
        torch_dtype=model_config.get("torch_dtype", "float16"),
        attn_implementation=model_config.get("attn_implementation", "eager"),
        max_memory=max_memory, offload_folder=str(offload_dir), offload_state_dict=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = get_process_model("llava")(model, tokenizer, image_processor)
    elapsed = time.monotonic() - started
    record_timing(f"{role} model load and adapter initialization", elapsed)
    report_timing(f"Load {role} model and initialize adapter", elapsed, enabled=args.timing)
    log(
        f"Loaded {role} checkpoint={checkpoint} in {elapsed:.1f}s; "
        f"placement={getattr(model, 'hf_device_map', 'single-device')}", level="extensive",
    )
    return model, adapter, tokenizer


def validate_compatibility(expert_tokenizer: Any, amateur_tokenizer: Any, expert_adapter: Any, amateur_adapter: Any) -> None:
    """Reject model pairs whose token or image processing spaces cannot be compared."""
    if expert_tokenizer.get_vocab() != amateur_tokenizer.get_vocab():
        raise ValueError("Expert and amateur tokenizers have different vocabularies; contrastive token scores cannot be aligned")
    special_names = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    different = [name for name in special_names if getattr(expert_tokenizer, name, None) != getattr(amateur_tokenizer, name, None)]
    if different:
        raise ValueError(f"Expert and amateur tokenizers disagree on special tokens: {', '.join(different)}")
    expert_processor = expert_adapter.image_processor
    amateur_processor = amateur_adapter.image_processor
    for name in ("crop_size", "size", "image_mean", "image_std"):
        if getattr(expert_processor, name, None) != getattr(amateur_processor, name, None):
            raise ValueError(f"Expert and amateur image processors differ in {name}")


def prepare_prompt(
    adapter: Any, image: Any, prompt_row: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return multimodal embeddings, attention mask, and readable model-formatted prompt."""
    prepared = adapter.preprocess_data([image], prompt_row)
    input_ids = prepared["input_ids"].detach().cpu().reshape(-1).tolist()
    text_parts: list[str] = []
    text_token_ids: list[int] = []
    for token_id in input_ids:
        if token_id < 0:
            if text_token_ids:
                text_parts.append(adapter.tokenizer.decode(
                    text_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
                ))
                text_token_ids.clear()
            text_parts.append("[IMAGE]")
        else:
            text_token_ids.append(token_id)
    if text_token_ids:
        text_parts.append(adapter.tokenizer.decode(
            text_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        ))
    prompt_text = "".join(text_parts)
    batch = adapter.data_collator([prepared])
    prompt_inputs, prompt_kwargs = adapter.generate_input(batch)
    return prompt_inputs["inputs_embeds"], prompt_kwargs["attention_mask"], prompt_text


def print_prompt_comparison(
    sample_index: int,
    image: Any,
    image_path: Path,
    expert_prompt: str | None,
    amateur_prompt: str | None,
) -> None:
    """Save the sample image and print both model-formatted prompts in two columns."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    log(f"Sample {sample_index} image saved to {image_path}", level="normal")

    columns = [
        ("EXPERT PROMPT", expert_prompt or "(not prepared for this run)"),
        ("AMATEUR PROMPT", amateur_prompt or "(not prepared for this run)"),
    ]
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    column_width = max(24, min(72, (terminal_width - 3) // 2))
    wrapped = [
        textwrap.wrap(" ".join(prompt.split()), width=column_width)
        or [""]
        for _, prompt in columns
    ]
    print(f"Sample {sample_index} prompts (image token shown as [IMAGE]):", flush=True)
    print(f"{columns[0][0]:<{column_width}} | {columns[1][0]}", flush=True)
    print(f"{'-' * column_width}-+-{'-' * column_width}", flush=True)
    for line_index in range(max(map(len, wrapped))):
        left = wrapped[0][line_index] if line_index < len(wrapped[0]) else ""
        right = wrapped[1][line_index] if line_index < len(wrapped[1]) else ""
        print(f"{left:<{column_width}} | {right}", flush=True)


def eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Resolve the model's EOS IDs to a set of integer token IDs."""
    value = getattr(model.config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    return set(value) if isinstance(value, (list, tuple)) else ({int(value)} if value is not None else set())


def decode_one(
    model: Any,
    tokenizer: Any,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new_tokens: int,
    *,
    amateur: tuple[Any, torch.Tensor, torch.Tensor] | None = None,
    contrastive_weight: float = 0.5,
    plausibility_threshold: float = 0.1,
    progress: Any = None,
) -> tuple[list[int], str]:
    """Greedily decode with one model or expert-minus-amateur token log scores."""
    embedding = model.get_input_embeddings()
    current_embeds = prompt_embeds
    attention = prompt_mask
    past = None
    amateur_model = amateur[0] if amateur is not None else None
    amateur_current = amateur[1] if amateur is not None else None
    amateur_attention = amateur[2] if amateur is not None else None
    amateur_past = None
    generated: list[int] = []
    eos_ids = eos_token_ids(model, tokenizer)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            expert_outputs = model(
                inputs_embeds=current_embeds, attention_mask=attention, past_key_values=past,
                use_cache=True, return_dict=True,
            )
            expert_logits = expert_outputs.logits[:, -1, :].float()
            if amateur_model is None:
                scores = expert_logits[0]
            else:
                amateur_outputs = amateur_model(
                    inputs_embeds=amateur_current, attention_mask=amateur_attention,
                    past_key_values=amateur_past, use_cache=True, return_dict=True,
                )
                amateur_logits = amateur_outputs.logits[:, -1, :].float()
                if expert_logits.shape[-1] != amateur_logits.shape[-1]:
                    raise RuntimeError("Expert and amateur output vocabulary sizes differ")
                expert_log_probs = torch.log_softmax(expert_logits[0], dim=-1)
                amateur_log_probs = torch.log_softmax(amateur_logits[0].to(expert_log_probs.device), dim=-1)
                expert_probs = expert_log_probs.exp()
                allowed = expert_probs >= (expert_probs.max() * plausibility_threshold)
                scores = (expert_log_probs - contrastive_weight * amateur_log_probs).masked_fill(~allowed, -torch.inf)

            next_id = int(scores.argmax().item())
            generated.append(next_id)
            if progress is not None and (step + 1) % 16 == 0:
                progress(step + 1)
            if next_id in eos_ids:
                break

            token = torch.tensor([[next_id]], dtype=torch.long, device=embedding.weight.device)
            next_embed = embedding(token)
            attention = torch.cat((attention, attention.new_ones((attention.shape[0], 1))), dim=1)
            past = expert_outputs.past_key_values
            current_embeds = next_embed if past is not None else torch.cat((current_embeds, next_embed), dim=1)

            if amateur_model is not None:
                amateur_embedding = amateur_model.get_input_embeddings()
                amateur_token = token.to(amateur_embedding.weight.device)
                amateur_next_embed = amateur_embedding(amateur_token)
                amateur_attention = torch.cat(
                    (amateur_attention, amateur_attention.new_ones((amateur_attention.shape[0], 1))), dim=1
                )
                amateur_past = amateur_outputs.past_key_values
                amateur_current = (
                    amateur_next_embed if amateur_past is not None
                    else torch.cat((amateur_current, amateur_next_embed), dim=1)
                )

    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return generated, text


def emit_timing_summary(started: float, *, enabled: bool) -> None:
    """Print measured operation totals when extensive verbosity is active."""
    if not enabled:
        return
    log("Timing summary (nested stage totals may overlap):", level="extensive")
    for stage, values in _TIMINGS.items():
        log(
            f"Timing {stage}: count={len(values)}, total={sum(values):.1f}s, "
            f"avg={sum(values) / len(values):.2f}s, min={min(values):.2f}s, max={max(values):.2f}s",
            level="extensive",
        )
    log(f"Timing total process wall time: {time.monotonic() - started:.1f}s", level="extensive")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    started = time.monotonic()
    timing = args.timing or args.log_level == "extensive"
    args.timing = timing
    try:
        if args.samples is not None and args.samples < 1:
            raise ValueError("--samples must be positive")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive")
        if not math.isfinite(args.contrastive_weight) or args.contrastive_weight < 0:
            raise ValueError("--contrastive-weight must be finite and non-negative")
        if not math.isfinite(args.plausibility_threshold) or not 0 < args.plausibility_threshold <= 1:
            raise ValueError("--plausibility-threshold must be in (0, 1]")
        if len(set(args.decoding_modes)) != len(args.decoding_modes):
            raise ValueError("--decoding-modes cannot repeat a mode")
        if "contrastive" in args.decoding_modes and "amateur" not in args.decoding_modes and not args.amateur_model:
            raise ValueError("--amateur-model is required for contrastive decoding")

        model_config = read_yaml(args.model_config)
        dataset_config = read_yaml(args.dataset_config)
        expert_checkpoint = resolve_path(args.expert_model or model_config.get("model_dir") or model_config.get("model_id"))
        amateur_checkpoint = resolve_path(args.amateur_model)
        device = args.device or model_config.get("device_map", "cuda:0")
        if device not in {"cpu", "cuda", "auto"} and not re.fullmatch(r"cuda:\d+", device):
            raise ValueError("--device must be cpu, auto, cuda, or a CUDA device such as cuda:0")
        sample_count = args.samples if args.samples is not None else dataset_config.get("calibration_samples", 2)
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("dataset calibration_samples must be a positive integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("dataset calibration_seed must be an integer or null")
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"
        metadata_path = output_dir / "run_config.json"
        if not args.overwrite and (results_path.exists() or metadata_path.exists()):
            raise FileExistsError(f"Results already exist under {output_dir}; pass --overwrite to replace them")

        # ====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # ====================================================================
        dataset_started = time.monotonic()
        source = dataset_config.get("calibration_source", "coco")
        dataset = load_sharegpt4v_dataset(
            source=source, config_path=args.dataset_config, download_images=False, existing_images_only=True,
        )
        if sample_count > len(dataset):
            raise ValueError(f"Requested {sample_count} samples but only {len(dataset)} records have local images")
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))
        dataset_elapsed = time.monotonic() - dataset_started
        record_timing("dataset loading and sample selection", dataset_elapsed)
        report_timing("Load dataset and select examples", dataset_elapsed, enabled=args.timing)
        log(f"Selected {len(dataset)} examples from source={source!r}, seed={seed}")

        # ====================================================================
        # 2. LOAD MODELS
        # ====================================================================
        qig_root_value = model_config.get("implementation_source_dir", "third_party/QIG")
        qig_root = resolve_path(qig_root_value)
        runtime_started = time.monotonic()
        builder, get_process_model = load_runtime(qig_root)
        runtime_elapsed = time.monotonic() - runtime_started
        record_timing("model runtime imports", runtime_elapsed)
        report_timing("Import model runtime", runtime_elapsed, enabled=args.timing)
        log(
            f"Expert={expert_checkpoint}; amateur={amateur_checkpoint}; device={device}; "
            f"modes={','.join(args.decoding_modes)}; samples={len(dataset)}; max_new_tokens={args.max_new_tokens}",
            level="normal",
        )
        need_expert = "expert" in args.decoding_modes or "contrastive" in args.decoding_modes
        expert = expert_adapter = expert_tokenizer = None
        if need_expert:
            expert, expert_adapter, expert_tokenizer = load_model(
                builder, get_process_model, expert_checkpoint, model_config, args, "expert", device, output_dir,
            )
        need_amateur = "amateur" in args.decoding_modes or "contrastive" in args.decoding_modes
        amateur = amateur_adapter = amateur_tokenizer = None
        if need_amateur:
            amateur, amateur_adapter, amateur_tokenizer = load_model(
                builder, get_process_model, amateur_checkpoint, model_config, args, "amateur", device, output_dir,
            )
            if "contrastive" in args.decoding_modes:
                validate_compatibility(expert_tokenizer, amateur_tokenizer, expert_adapter, amateur_adapter)
                log("Expert and amateur tokenizers and image processors are compatible", level="extensive")

        # ====================================================================
        # 3. REGISTER HOOKS
        # ====================================================================
        # This generation comparison needs no capture hooks; model-level forward hooks would add overhead.
        log("No capture hooks required for token-level decoding comparison", level="extensive")

        # ====================================================================
        # 4. FORWARD CALIBRATION SAMPLES
        # ====================================================================
        with results_path.open("w", encoding="utf-8") as result_file:
            for sample_index, sample in enumerate(dataset, start=1):
                conversations = sample["conversations"]
                user_indices = [i for i, turn in enumerate(conversations) if turn.get("from") in {"human", "user"}]
                if not user_indices:
                    raise ValueError(f"Sample {sample.get('id', sample_index)} has no user turn")
                prompt_row = {
                    "conversations": conversations[: user_indices[-1] + 1],
                    "id": str(sample.get("id", sample_index)), "image": "local",
                }
                image = sample["image"].convert("RGB") if hasattr(sample["image"], "convert") else sample["image"]
                outputs: dict[str, dict[str, Any]] = {}
                modes_started = time.monotonic()
                needed_roles = set()
                if "expert" in args.decoding_modes or "contrastive" in args.decoding_modes:
                    needed_roles.add("expert")
                if "amateur" in args.decoding_modes or "contrastive" in args.decoding_modes:
                    needed_roles.add("amateur")
                prepared_prompts: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
                for role in ("expert", "amateur"):
                    if role not in needed_roles:
                        continue
                    role_adapter = expert_adapter if role == "expert" else amateur_adapter
                    prep_started = time.monotonic()
                    prompt_embeds, prompt_mask, prompt_text = prepare_prompt(role_adapter, image, prompt_row)
                    if args.timing and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prep_elapsed = time.monotonic() - prep_started
                    record_timing(f"{role} sample preprocessing", prep_elapsed)
                    report_timing(f"Prepare {role} prompt", prep_elapsed, enabled=args.timing)
                    log(
                        f"Prepared {role} prompt embeddings={tuple(prompt_embeds.shape)} "
                        f"dtype={prompt_embeds.dtype} device={prompt_embeds.device}; "
                        f"attention_mask={tuple(prompt_mask.shape)} device={prompt_mask.device}",
                        level="extensive",
                    )
                    prepared_prompts[role] = (prompt_embeds, prompt_mask, prompt_text)

                if args.preview_inputs:
                    print_prompt_comparison(
                        sample_index,
                        image,
                        output_dir / f"sample-{sample_index:04d}.png",
                        prepared_prompts.get("expert", (None, None, None))[2],
                        prepared_prompts.get("amateur", (None, None, None))[2],
                    )

                for mode in args.decoding_modes:
                    model = amateur if mode == "amateur" else expert
                    prompt_embeds, prompt_mask, _ = prepared_prompts[
                        "amateur" if mode == "amateur" else "expert"
                    ]
                    if mode == "contrastive":
                        amateur_embeds, amateur_mask, _ = prepared_prompts["amateur"]
                    forward_started = time.monotonic()
                    progress = lambda count: log(
                        f"Sample {sample_index}/{len(dataset)} mode={mode} generated "
                        f"{count}/{args.max_new_tokens} tokens"
                    )
                    if mode == "contrastive":
                        token_ids, answer = decode_one(
                            expert, expert_tokenizer, prompt_embeds, prompt_mask, args.max_new_tokens,
                            amateur=(amateur, amateur_embeds, amateur_mask),
                            contrastive_weight=args.contrastive_weight,
                            plausibility_threshold=args.plausibility_threshold,
                            progress=progress,
                        )
                    else:
                        token_ids, answer = decode_one(
                            model, amateur_tokenizer if mode == "amateur" else expert_tokenizer,
                            prompt_embeds, prompt_mask, args.max_new_tokens, progress=progress,
                        )
                    if args.timing and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    generation_elapsed = time.monotonic() - forward_started
                    record_timing(f"{mode} generation", generation_elapsed)
                    report_timing(f"Generate {mode} answer", generation_elapsed, enabled=args.timing)
                    outputs[mode] = {"token_ids": token_ids, "text": answer}
                    print(
                        f"\n[Sample {sample_index}/{len(dataset)} | {mode} answer]\n{answer}\n",
                        flush=True,
                    )
                    log(
                        f"Sample {sample_index}/{len(dataset)} mode={mode} generated {len(token_ids)} tokens; "
                        f"elapsed={time.monotonic() - modes_started:.1f}s",
                        level="normal",
                    )
                del prepared_prompts
                result = {
                    "sample_id": str(sample.get("id", sample_index)),
                    "prompt": prompt_row["conversations"],
                    "modes": outputs,
                }
                write_started = time.monotonic()
                result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                result_file.flush()
                write_elapsed = time.monotonic() - write_started
                record_timing("JSONL result write and flush", write_elapsed)
                report_timing("Write and flush result", write_elapsed, enabled=args.timing)
                log(f"Sample {sample_index}/{len(dataset)} saved", level="extensive")

        # ====================================================================
        # 5. ANALYSE DECODED OUTPUTS
        # ====================================================================
        metadata = {
            "expert_model": str(expert_checkpoint), "amateur_model": str(amateur_checkpoint),
            "modes": list(args.decoding_modes), "samples": len(dataset), "seed": seed,
            "contrastive_weight": args.contrastive_weight,
            "plausibility_threshold": args.plausibility_threshold,
            "max_new_tokens": args.max_new_tokens, "device": device,
            "expert_memory_limits_bytes": {
                "gpu": args.max_expert_model_memory_gpu, "cpu": args.max_expert_model_memory_cpu,
            },
            "amateur_memory_limits_bytes": {
                "gpu": args.max_amateur_model_memory_gpu, "cpu": args.max_amateur_model_memory_cpu,
            },
            "expert_device_map": getattr(expert, "hf_device_map", None) if expert is not None else None,
            "amateur_device_map": getattr(amateur, "hf_device_map", None) if amateur is not None else None,
        }
        metadata_started = time.monotonic()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        metadata_elapsed = time.monotonic() - metadata_started
        record_timing("run metadata serialization and save", metadata_elapsed)
        report_timing("Write run metadata", metadata_elapsed, enabled=args.timing)
        log(f"Saved decoded comparisons to {results_path}; run settings: {metadata_path}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        emit_timing_summary(started, enabled=args.log_level == "extensive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
