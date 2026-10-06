"""Compare a QIG checkpoint with its base model on the saved calibration set."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import get_data_paths  # noqa: E402
from attention_quantization.models.qig_loader import (  # noqa: E402
    load_qig_llava_model,
    load_qig_quantized_model,
)
from attention_quantization.quantization.qig import load_qig_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure layer-wise decoder activation error and final-logit KL divergence "
            "between a QIG checkpoint and its base LLaVA model."
        )
    )
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=REPOSITORY_ROOT / "configs" / "model.yaml")
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--samples", type=int, default=None, help="Use the first N records from the quantization calibration JSONL.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of calibration samples per forward pass.")
    parser.add_argument(
        "--sequential-model-loading",
        action="store_true",
        help="Reload base and quantized checkpoints for each batch to minimize retained host memory; slower.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Metrics JSON path (default: <quantized-model>/evaluation_metrics.json).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kl-token-chunk-size", type=int, default=8, help="Limit temporary memory while computing KL over vocabulary logits.")
    return parser.parse_args()


def make_process_model(runtime: Any, lm: Any) -> Any:
    process_class = runtime.get_process_model("llava")
    return process_class(
        lm._model,
        lm._tokenizer,
        getattr(lm, "_image_processor", None),
    )


def decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find LLaVA decoder layers at model.model.layers")
    return list(layers)


def capture_forward(
    process_model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    """Forward a batch, retaining each sample's decoder outputs and logits on CPU."""
    model = process_model.model
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(decoder_layers(model)):
        def save_output(_module: Any, _inputs: Any, output: Any, *, layer_index: int = index) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                captured[layer_index] = hidden.detach().to(device="cpu", dtype=torch.float16)

        handles.append(layer.register_forward_hook(save_output))

    try:
        device = next(model.parameters()).device
        with torch.inference_mode():
            output = process_model(
                inputs_embeds=inputs_embeds.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
                use_cache=False,
                return_dict=True,
            )
        batch_size = inputs_embeds.shape[0]
        activations = [
            [captured[layer_index][sample_index] for layer_index in range(len(handles))]
            for sample_index in range(batch_size)
        ]
        logits = [
            output.logits.detach().to(device="cpu", dtype=torch.float16)[sample_index]
            for sample_index in range(batch_size)
        ]
    finally:
        for handle in handles:
            handle.remove()
    return activations, logits


def prepare_batch_inputs(
    process_model: Any,
    records: list[dict[str, Any]],
    raw_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prepared_records = []
    for record in records:
        image_path = (raw_dir / record["image"]).resolve()
        try:
            image_path.relative_to(raw_dir.resolve())
        except ValueError as error:
            raise ValueError(f"Calibration image path escapes configured raw directory: {image_path}") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"Calibration image is missing: {image_path}")
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        prepared_records.append(process_model.preprocess_data([image], record))
    batch = process_model.data_collator(prepared_records)
    prompt_inputs, prompt_kwargs = process_model.generate_input(batch)
    return (
        prompt_inputs["inputs_embeds"].detach().cpu(),
        prompt_kwargs["attention_mask"].detach().cpu(),
        prompt_kwargs["labels"].detach().cpu(),
    )


def compute_layer_errors(
    baseline: list[list[torch.Tensor]],
    quantized: list[list[torch.Tensor]],
    attention_masks: list[torch.Tensor],
) -> list[dict[str, float | int]]:
    if not baseline or len(baseline) != len(quantized):
        raise ValueError("Baseline and quantized activation collections do not match")
    result = []
    for layer_index, (base_samples, quant_samples) in enumerate(zip(baseline, quantized)):
        squared_error = 0.0
        squared_reference = 0.0
        token_count = 0
        for base, quant, mask in zip(base_samples, quant_samples, attention_masks):
            if base.shape != quant.shape:
                raise ValueError(f"Layer {layer_index} activation shapes differ: {base.shape} vs {quant.shape}")
            valid = mask[0].bool()
            base_values = base[valid].float()
            difference = base_values - quant[valid].float()
            squared_error += difference.square().sum().item()
            squared_reference += base_values.square().sum().item()
            token_count += int(valid.sum())
        result.append(
            {
                "layer": layer_index,
                "relative_l2_error": (squared_error / max(squared_reference, 1e-12)) ** 0.5,
                "rmse": (squared_error / max(token_count, 1)) ** 0.5,
                "valid_tokens": token_count,
            }
        )
    return result


def compute_final_kl(
    baseline_logits: list[torch.Tensor],
    quantized_logits: list[torch.Tensor],
    labels: list[torch.Tensor],
    chunk_size: int,
    device: torch.device,
) -> tuple[float, int]:
    """Compute mean KL(base || quantized) over assistant-answer next-token positions."""
    total_kl = 0.0
    total_tokens = 0
    for base, quant, target_labels in zip(baseline_logits, quantized_logits, labels):
        if base.shape != quant.shape:
            raise ValueError(f"Final logit shapes differ: {base.shape} vs {quant.shape}")
        # At position t, causal logits predict token t+1. Labels mark assistant tokens.
        positions = (target_labels[0, 1:] != -100).nonzero(as_tuple=True)[0]
        for start in range(0, positions.numel(), chunk_size):
            selected = positions[start : start + chunk_size]
            base_logit_chunk = base[selected].to(device=device, dtype=torch.float32)
            quant_logit_chunk = quant[selected].to(device=device, dtype=torch.float32)
            base_log_probs = functional.log_softmax(base_logit_chunk, dim=-1)
            quant_log_probs = functional.log_softmax(quant_logit_chunk, dim=-1)
            base_probs = base_log_probs.exp()
            token_kl = (base_probs * (base_log_probs - quant_log_probs)).sum(dim=-1)
            total_kl += token_kl.sum().item()
            total_tokens += int(selected.numel())
    return total_kl / max(total_tokens, 1), total_tokens


def release_model() -> None:
    """Collect released model references and return unused CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_sequential_batches(
    *,
    records: list[dict[str, Any]],
    batch_size: int,
    base_model: Path,
    quantized_dir: Path,
    qig_source: Path,
    runtime: Any,
    raw_dir: Path,
    device: str,
    attn_implementation: str,
    kl_chunk_size: int,
) -> tuple[list[dict[str, float | int]], float, int]:
    """Compare each batch immediately, holding only one model and batch at a time."""
    layer_stats: list[list[float | int]] | None = None
    total_kl = 0.0
    total_kl_tokens = 0
    batches = [records[start : start + batch_size] for start in range(0, len(records), batch_size)]

    for batch_index, record_batch in enumerate(batches, start=1):
        print(f"Loading base model for batch {batch_index}/{len(batches)}", flush=True)
        baseline_lm, baseline_adapter = load_qig_llava_model(
            base_model,
            qig_source_dir=qig_source,
            device=device,
            attn_implementation=attn_implementation,
        )
        batch_input = prepare_batch_inputs(baseline_adapter, record_batch, raw_dir)
        baseline_activations, baseline_logits = capture_forward(baseline_adapter, *batch_input)
        del baseline_adapter, baseline_lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading quantized model for batch {batch_index}/{len(batches)}", flush=True)
        quantized_lm = load_qig_quantized_model(
            quantized_dir,
            device=device,
            qig_source_dir=qig_source,
        )
        quantized_adapter = make_process_model(runtime, quantized_lm)
        quantized_activations, quantized_logits = capture_forward(quantized_adapter, *batch_input)
        if not baseline_activations or len(baseline_activations[0]) != len(quantized_activations[0]):
            raise ValueError("Base and quantized model decoder layer counts differ")

        base_by_layer = [
            [sample[layer_index] for sample in baseline_activations]
            for layer_index in range(len(baseline_activations[0]))
        ]
        quant_by_layer = [
            [sample[layer_index] for sample in quantized_activations]
            for layer_index in range(len(quantized_activations[0]))
        ]
        batch_masks = [batch_input[1][i : i + 1] for i in range(len(record_batch))]
        batch_labels = [batch_input[2][i : i + 1] for i in range(len(record_batch))]
        batch_layer_errors = compute_layer_errors(base_by_layer, quant_by_layer, batch_masks)
        if layer_stats is None:
            layer_stats = [[0.0, 0.0, 0] for _ in batch_layer_errors]
        for stats, error in zip(layer_stats, batch_layer_errors):
            count = int(error["valid_tokens"])
            squared_error = float(error["rmse"]) ** 2 * count
            relative_error = float(error["relative_l2_error"])
            squared_reference = squared_error / max(relative_error**2, 1e-24)
            stats[0] = float(stats[0]) + squared_error
            stats[1] = float(stats[1]) + squared_reference
            stats[2] = int(stats[2]) + count

        batch_kl, batch_kl_tokens = compute_final_kl(
            baseline_logits,
            quantized_logits,
            batch_labels,
            kl_chunk_size,
            next(quantized_adapter.model.parameters()).device,
        )
        total_kl += batch_kl * batch_kl_tokens
        total_kl_tokens += batch_kl_tokens
        print(f"Compared batch {batch_index}/{len(batches)}", flush=True)

        del quantized_adapter, quantized_lm
        del baseline_activations, baseline_logits, quantized_activations, quantized_logits, batch_input
        del baseline_adapter, baseline_lm
        release_model()

    if layer_stats is None:
        raise ValueError("No evaluation batches were processed")
    layer_errors = [
        {
            "layer": index,
            "relative_l2_error": (float(stats[0]) / max(float(stats[1]), 1e-12)) ** 0.5,
            "rmse": (float(stats[0]) / max(int(stats[2]), 1)) ** 0.5,
            "valid_tokens": int(stats[2]),
        }
        for index, stats in enumerate(layer_stats)
    ]
    return layer_errors, total_kl / max(total_kl_tokens, 1), total_kl_tokens


def main() -> int:
    args = parse_args()
    try:
        quantized_dir = args.quantized_model.expanduser().resolve()
        metadata_path = quantized_dir / "qig_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"QIG metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        calibration_name = metadata.get("calibration_jsonl")
        if not calibration_name:
            raise ValueError("The checkpoint metadata does not reference a calibration JSONL")
        calibration_path = quantized_dir / calibration_name
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Saved calibration records not found: {calibration_path}")
        records = [json.loads(line) for line in calibration_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sample_count = args.samples if args.samples is not None else len(records)
        if sample_count < 1 or sample_count > len(records):
            raise ValueError(f"--samples must be between 1 and {len(records)}")
        records = records[:sample_count]
        if args.kl_token_chunk_size < 1:
            raise ValueError("--kl-token-chunk-size must be positive")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")

        model_config = read_yaml(args.model_config)
        dataset_paths = get_data_paths(args.dataset_config)
        configured_base = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        base_model = args.base_model.expanduser().resolve() if args.base_model else configured_base
        if not base_model.is_absolute():
            base_model = (REPOSITORY_ROOT / base_model).resolve()
        qig_source = model_config.get("implementation_source_dir", "third_party/QIG")
        qig_source = Path(qig_source).expanduser()
        if not qig_source.is_absolute():
            qig_source = (REPOSITORY_ROOT / qig_source).resolve()
        runtime = load_qig_runtime(qig_source)

        started = time.perf_counter()
        if args.sequential_model_loading:
            print(
                "Sequential mode: loading and releasing each model for every batch; expect slower evaluation.",
                flush=True,
            )
            layer_errors, mean_kl, kl_tokens = evaluate_sequential_batches(
                records=records,
                batch_size=args.batch_size,
                base_model=base_model,
                quantized_dir=quantized_dir,
                qig_source=qig_source,
                runtime=runtime,
                raw_dir=dataset_paths["raw_dir"],
                device=args.device,
                attn_implementation=model_config.get("attn_implementation", "eager"),
                kl_chunk_size=args.kl_token_chunk_size,
            )
        else:
            baseline_activations: list[list[torch.Tensor]] = []
            baseline_logits: list[torch.Tensor] = []
            input_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            attention_masks: list[torch.Tensor] = []
            target_labels: list[torch.Tensor] = []

            print(f"Loading base model: {base_model}", flush=True)
            baseline_lm, baseline_adapter = load_qig_llava_model(
                base_model,
                qig_source_dir=qig_source,
                device=args.device,
                attn_implementation=model_config.get("attn_implementation", "eager"),
            )
            batches = [records[start : start + args.batch_size] for start in range(0, sample_count, args.batch_size)]
            for batch_index, record_batch in enumerate(batches, start=1):
                batch_input = prepare_batch_inputs(baseline_adapter, record_batch, dataset_paths["raw_dir"])
                input_batches.append(batch_input)
                batch_activations, batch_logits = capture_forward(baseline_adapter, *batch_input)
                if not baseline_activations:
                    baseline_activations = [[] for _ in batch_activations[0]]
                for sample_activations in batch_activations:
                    for layer_index, activation in enumerate(sample_activations):
                        baseline_activations[layer_index].append(activation)
                baseline_logits.extend(batch_logits)
                attention_masks.extend(batch_input[1][i : i + 1] for i in range(len(record_batch)))
                target_labels.extend(batch_input[2][i : i + 1] for i in range(len(record_batch)))
                print(f"Base batch {batch_index}/{len(batches)} complete ({len(record_batch)} samples)", flush=True)
            layer_count = len(baseline_activations)
            del baseline_adapter, baseline_lm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Loading quantized model: {quantized_dir}", flush=True)
            quantized_lm = load_qig_quantized_model(
                quantized_dir,
                device=args.device,
                qig_source_dir=qig_source,
            )
            quantized_adapter = make_process_model(runtime, quantized_lm)
            quantized_activations: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
            quantized_logits: list[torch.Tensor] = []
            for batch_index, batch_input in enumerate(input_batches, start=1):
                batch_activations, batch_logits = capture_forward(quantized_adapter, *batch_input)
                if batch_activations and len(batch_activations[0]) != layer_count:
                    raise ValueError(f"Decoder layer count differs: base={layer_count}, quantized={len(batch_activations[0])}")
                for sample_activations in batch_activations:
                    for layer_index, activation in enumerate(sample_activations):
                        quantized_activations[layer_index].append(activation)
                quantized_logits.extend(batch_logits)
                print(f"Quantized batch {batch_index}/{len(input_batches)} complete", flush=True)

            layer_errors = compute_layer_errors(baseline_activations, quantized_activations, attention_masks)
            mean_kl, kl_tokens = compute_final_kl(
                baseline_logits,
                quantized_logits,
                target_labels,
                args.kl_token_chunk_size,
                next(quantized_adapter.model.parameters()).device,
            )
        if kl_tokens == 0:
            raise ValueError("No assistant-answer tokens were found in the saved calibration records")
        result = {
            "base_model": str(base_model),
            "quantized_model": str(quantized_dir),
            "samples": sample_count,
            "batch_size": args.batch_size,
            "sequential_model_loading": args.sequential_model_loading,
            "calibration_seed": metadata.get("calibration_seed"),
            "layer_error_definition": "relative L2 and RMSE of decoder block outputs over non-padding tokens",
            "layers": layer_errors,
            "final_kl_definition": "mean KL(base || quantized) over assistant-answer next-token positions, in nats",
            "final_kl_divergence": mean_kl,
            "kl_token_count": kl_tokens,
            "elapsed_seconds": time.perf_counter() - started,
        }
        output_path = args.output.expanduser() if args.output else quantized_dir / "evaluation_metrics.json"
        if not output_path.is_absolute():
            output_path = (REPOSITORY_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("\nAverage decoder layer-wise error (relative L2):", flush=True)
        for entry in layer_errors:
            print(f"  Layer {entry['layer']:>2}: {entry['relative_l2_error']:.6f} (RMSE {entry['rmse']:.6f})")
        print(f"Final KL divergence, base || quantized: {mean_kl:.6f} nats/token ({kl_tokens} tokens)")
        print(f"Metrics saved to: {output_path}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
