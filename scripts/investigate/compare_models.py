"""Compare two LLaVA checkpoints independently on identical multimodal prompts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from attention_quantization.config import read_yaml  # noqa: E402
from attention_quantization.data import load_sharegpt4v_dataset  # noqa: E402
from contrastive_decoding import (  # noqa: E402
    decode_one,
    emit_timing_summary,
    load_model,
    load_runtime,
    parse_memory_size,
    prepare_prompt,
    print_prompt_comparison,
    record_timing,
    report_timing,
    resolve_path,
    wrap_display_text,
)
from investigation_logging import configure_logging, log  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read checkpoint, sampling, placement, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset.yaml")
    parser.add_argument("--model-a", type=Path, help="First local checkpoint; defaults to model.yaml.")
    parser.add_argument("--model-b", type=Path, help="Second local checkpoint; required unless saving images only.")
    parser.add_argument("--samples", type=int, help="Override dataset.yaml calibration_samples.")
    parser.add_argument("--seed", type=int, help="Override dataset.yaml calibration_seed; also seeds generation.")
    parser.add_argument(
        "--generation-modes", nargs="+", choices=("greedy", "probabilistic"), default=("greedy",),
        help="Generation strategies to run for both models (default: greedy).",
    )
    parser.add_argument("--num-generations", type=int, default=1, help="Responses per probabilistic setting.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Probabilistic sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability threshold.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens per response.")
    parser.add_argument(
        "--preview-inputs", action=argparse.BooleanOptionalAction, default=True,
        help="Save sample images and show model-formatted prompts (default: on).",
    )
    parser.add_argument("--display-images", action="store_true", help="Display each image with matplotlib.")
    parser.add_argument("--save-images-only", action="store_true", help="Save sample images without loading models.")
    parser.add_argument("--device", help="Device for all weights; defaults to model.yaml (cuda:0).")
    for role in ("a", "b"):
        for tier in ("gpu", "cpu"):
            parser.add_argument(
                f"--max-model-{role}-memory-{tier}", type=parse_memory_size,
                # Match the role-specific namespace expected by the shared loader.
                dest=f"max_model_{role}_model_memory_{tier}",
                help=f"Model {role.upper()} {tier.upper()} weight-placement limit; bare numbers mean GB.",
            )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/investigations/compare-models",
        help="Directory for results.jsonl, run_config.json, and preview images.",
    )
    parser.add_argument("--log-level", choices=("none", "normal", "extensive"), default="normal")
    parser.add_argument("--timing", action="store_true", help="Show operation timing at normal verbosity.")
    parser.add_argument("--quiet-warnings", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing results and run metadata.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid settings before loading datasets or model weights."""
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.save_images_only:
        return
    if args.model_b is None:
        raise ValueError("--model-b is required to compare two checkpoints")
    if args.device == "auto":
        raise ValueError("--device auto permits CPU placement; choose cpu or a specific CUDA device")
    if args.max_new_tokens < 1 or args.num_generations < 1:
        raise ValueError("--max-new-tokens and --num-generations must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if len(set(args.generation_modes)) != len(args.generation_modes):
        raise ValueError("--generation-modes cannot repeat a mode")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    started = time.monotonic()
    try:
        validate_args(args)
        dataset_config = read_yaml(args.dataset_config)
        sample_count = args.samples if args.samples is not None else dataset_config.get("calibration_samples", 2)
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("dataset calibration_samples must be a positive integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("dataset calibration_seed must be an integer or null")
        output_dir = resolve_path(args.output_dir)
        results_path = output_dir / "results.jsonl"
        metadata_path = output_dir / "run_config.json"
        if not args.save_images_only and not args.overwrite and (results_path.exists() or metadata_path.exists()):
            raise FileExistsError(f"Results already exist under {output_dir}; pass --overwrite to replace them")

        # ====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # ====================================================================
        stage_started = time.monotonic()
        source = dataset_config.get("calibration_source", "coco")
        dataset = load_sharegpt4v_dataset(
            source=source, config_path=args.dataset_config,
            download_images=False, existing_images_only=True,
        )
        if sample_count > len(dataset):
            raise ValueError(f"Requested {sample_count} samples but only {len(dataset)} records have local images")
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))
        elapsed = time.monotonic() - stage_started
        record_timing("dataset loading and sample selection", elapsed)
        report_timing("Load dataset and select examples", elapsed, enabled=args.timing)
        log(f"Selected {len(dataset)} examples from source={source!r}, seed={seed}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_images_only:
            for index, sample in enumerate(dataset, start=1):
                path = output_dir / f"sample-{index:04d}.png"
                sample["image"].convert("RGB").save(path)
                log(f"Saved sample {index}/{len(dataset)} image: {path}")
            return 0

        # ====================================================================
        # 2. LOAD MODELS
        # ====================================================================
        import torch

        model_config = read_yaml(args.model_config)
        first_checkpoint = args.model_a or model_config.get("model_dir") or model_config.get("model_id")
        if not first_checkpoint:
            raise ValueError("Pass --model-a or configure model_dir in model.yaml")
        checkpoints = {"model_a": resolve_path(first_checkpoint), "model_b": resolve_path(args.model_b)}
        device = args.device or model_config.get("device_map", "cuda:0")
        if device not in {"cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", device):
            raise ValueError("--device must be cpu, cuda, or a CUDA device such as cuda:0")
        if device.startswith("cuda"):
            gpu_index = int(device.split(":", 1)[1]) if ":" in device else 0
            strict_device_map: Any = {"": gpu_index}
        else:
            strict_device_map = "cpu"
        # Validate both paths before loading either large checkpoint.
        for checkpoint in checkpoints.values():
            if not (checkpoint / "config.json").is_file():
                raise FileNotFoundError(f"Model config was not found: {checkpoint / 'config.json'}")
        stage_started = time.monotonic()
        builder, get_process_model = load_runtime(resolve_path(model_config.get("implementation_source_dir", "third_party/QIG")))
        elapsed = time.monotonic() - stage_started
        record_timing("model runtime imports", elapsed)
        report_timing("Import model runtime", elapsed, enabled=args.timing)
        # Role -> (model, adapter, tokenizer); each checkpoint uses its own processors.
        models: dict[str, tuple[Any, Any, Any]] = {}
        for role, checkpoint in checkpoints.items():
            log(f"Loading {role}: {checkpoint}")
            models[role] = load_model(
                builder, get_process_model, checkpoint, model_config, args, role, device, output_dir,
                device_map=strict_device_map,
            )
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ====================================================================
        # 3. REGISTER HOOKS
        # ====================================================================
        log("No capture hooks required for independent generation", level="extensive")

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
                    "conversations": conversations[:user_indices[-1] + 1],
                    "id": str(sample.get("id", sample_index)), "image": "local",
                }
                image = sample["image"].convert("RGB")
                if args.display_images:
                    import matplotlib.pyplot as plt

                    plt.imshow(image)
                    plt.axis("off")
                    plt.title(f"Sample {sample_index}/{len(dataset)}")
                    plt.show()
                # Per-sample role -> prompt tensors/text and independently generated answers.
                prompts: dict[str, tuple[Any, Any, str]] = {}
                outputs: dict[str, list[dict[str, Any]]] = {}
                for role, (_, adapter, _) in models.items():
                    stage_started = time.monotonic()
                    prompts[role] = prepare_prompt(adapter, image, prompt_row)
                    if args.timing and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.monotonic() - stage_started
                    record_timing(f"{role} sample preprocessing", elapsed)
                    report_timing(f"Prepare {role} prompt", elapsed, enabled=args.timing)
                if args.preview_inputs:
                    print_prompt_comparison(
                        sample_index, image, output_dir / f"sample-{sample_index:04d}.png",
                        prompts["model_a"][2], prompts["model_b"][2],
                        labels=("MODEL A PROMPT", "MODEL B PROMPT"),
                    )
                for role, (model, _, tokenizer) in models.items():
                    prompt_embeds, prompt_mask, _ = prompts[role]
                    outputs[role] = []
                    for generation_mode in args.generation_modes:
                        count = args.num_generations if generation_mode == "probabilistic" else 1
                        for generation_index in range(1, count + 1):
                            log(f"Sample {sample_index}/{len(dataset)} {role} {generation_mode} #{generation_index}")
                            stage_started = time.monotonic()
                            token_ids, answer = decode_one(
                                model, tokenizer, prompt_embeds, prompt_mask, args.max_new_tokens,
                                generation_mode=generation_mode, temperature=args.temperature, top_p=args.top_p,
                                progress=lambda n: log(f"{role}: generated {n}/{args.max_new_tokens} tokens"),
                            )
                            if args.timing and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            elapsed = time.monotonic() - stage_started
                            record_timing(f"{role} {generation_mode} generation", elapsed)
                            report_timing(f"Generate {role} {generation_mode} answer", elapsed, enabled=args.timing)
                            outputs[role].append({
                                "generation_mode": generation_mode, "generation_index": generation_index,
                                "token_ids": token_ids, "text": answer,
                            })
                            width = shutil.get_terminal_size(fallback=(100, 24)).columns
                            print(
                                f"\n[Sample {sample_index}/{len(dataset)} | {role} | "
                                f"{generation_mode} #{generation_index}]\n{wrap_display_text(answer, width)}\n",
                                flush=True,
                            )
                del prompts, prompt_embeds, prompt_mask
                stage_started = time.monotonic()
                result_file.write(json.dumps({
                    "sample_id": prompt_row["id"], "prompt": prompt_row["conversations"], "models": outputs,
                }, ensure_ascii=False) + "\n")
                result_file.flush()
                elapsed = time.monotonic() - stage_started
                record_timing("JSONL result write and flush", elapsed)
                report_timing("Write and flush result", elapsed, enabled=args.timing)

        # ====================================================================
        # 5. ANALYSE DECODED OUTPUTS
        # ====================================================================
        metadata = {
            **{role: str(checkpoint) for role, checkpoint in checkpoints.items()},
            "model_config": str(resolve_path(args.model_config)),
            "dataset_config": str(resolve_path(args.dataset_config)),
            "generation_modes": list(args.generation_modes),
            "num_probabilistic_generations": args.num_generations,
            "temperature": args.temperature, "top_p": args.top_p,
            "samples": len(dataset), "seed": seed, "source": source,
            "max_new_tokens": args.max_new_tokens, "device": device,
        }
        for role, (model, _, _) in models.items():
            metadata[f"{role}_memory_limits_bytes"] = {
                tier: getattr(args, f"max_{role}_model_memory_{tier}") for tier in ("gpu", "cpu")
            }
            metadata[f"{role}_device_map"] = getattr(model, "hf_device_map", None)
        stage_started = time.monotonic()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        elapsed = time.monotonic() - stage_started
        record_timing("run metadata serialization and save", elapsed)
        report_timing("Write run metadata", elapsed, enabled=args.timing)
        log(f"Saved model comparisons to {results_path}; run settings: {metadata_path}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        emit_timing_summary(started, enabled=args.log_level == "extensive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
