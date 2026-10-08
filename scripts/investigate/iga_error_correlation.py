"""Correlate IGA-selected token errors with the quantized model's CE loss."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import get_data_paths, load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.models.qig_loader import (  # noqa: E402
    load_qig_llava_model,
    load_qig_quantized_model,
)
from attention_quantization.quantization.qig import load_qig_runtime  # noqa: E402
from investigation_logging import configure_logging, log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-layer output MSE on top-IGA image tokens and all valid tokens, "
            "then correlate both with per-sample final cross-entropy."
        )
    )
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=REPOSITORY_ROOT / "configs" / "model.yaml")
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--samples", type=int, default=128, help="Number of ShareGPT4V COCO samples to analyze.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for choosing the sample set.")
    parser.add_argument("--batch-size", type=int, default=1, help="Forward batch size; increase if GPU memory permits.")
    parser.add_argument("--top-percent", type=float, default=10.0, help="Image-token percentage selected by IGA.")
    parser.add_argument(
        "--correlation-target",
        choices=("kl", "ce_delta"),
        default="kl",
        help="Per-sample target: KL(base || quantized), or CE change (quantized CE - base CE).",
    )
    parser.add_argument(
        "--log-level",
        choices=("none", "normal", "extensive"),
        default="normal",
        help="Output detail: none, concise progress, or detailed progress and timings.",
    )
    parser.add_argument(
        "--quiet-warnings",
        action="store_true",
        help="Suppress Python and Hugging Face warning messages.",
    )
    parser.add_argument("--timing", action="store_true", help="Print elapsed times for data, model, and batch stages.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for result CSV and JSON files.")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_process_model(runtime: Any, lm: Any) -> Any:
    process_class = runtime.get_process_model("llava")
    return process_class(lm._model, lm._tokenizer, getattr(lm, "_image_processor", None))


def select_samples(config_path: Path, count: int, seed: int | None) -> Dataset:
    dataset = load_sharegpt4v_dataset(
        source="coco",
        config_path=config_path,
        download_images=False,
        existing_images_only=True,
    )
    if count > len(dataset):
        raise ValueError(
            f"Requested {count} samples, but only {len(dataset)} ShareGPT4V COCO records "
            "have images available locally"
        )
    if seed is not None or count < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(count))
    return dataset


def prepare_batch(adapter: Any, samples: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prepared_samples = []
    for index, sample in enumerate(samples):
        image = sample["image"]
        # The LLaVA pad preprocessor creates a canvas using an RGB background
        # color. Normalize grayscale/palette/RGBA inputs to RGB first.
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        row = {
            "conversations": sample["conversations"],
            "id": str(sample.get("id", index)),
            "image": "local",
        }
        prepared_samples.append(adapter.preprocess_data([image], row))
    batch = adapter.data_collator(prepared_samples)
    prompt_inputs, prompt_kwargs = adapter.generate_input(batch)
    tensors = {
        key: value.detach().cpu()
        for key, value in prompt_kwargs.items()
        if torch.is_tensor(value)
    }
    return prompt_inputs["inputs_embeds"].detach().cpu(), tensors


def image_and_text_masks(kwargs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    # Normalize adapter masks before boolean operations: some preprocessors
    # return attention_mask and vision_mask on different devices.
    attention_mask = kwargs["attention_mask"].bool().detach().cpu()
    image_mask = kwargs.get("vision_mask")
    if image_mask is None:
        raise ValueError("QIG input adapter did not provide a vision_mask for image-token selection")
    image_mask = image_mask.bool().detach().cpu() & attention_mask
    text_mask = attention_mask & ~image_mask
    for sample_index in range(image_mask.shape[0]):
        image_positions = image_mask[sample_index].nonzero(as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError(f"Sample {sample_index} has no image tokens")
        # As in investigate/investigate_attention.py, use text query tokens following the image.
        text_mask[sample_index] &= torch.arange(text_mask.shape[1]) > image_positions[-1]
        if not text_mask[sample_index].any():
            raise ValueError(f"Sample {sample_index} has no text tokens after the image")
    return image_mask, text_mask


def capture_layer_outputs_and_iga(
    adapter: Any,
    inputs_embeds: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    *,
    capture_iga: bool,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]] | None, torch.Tensor]:
    model = adapter.model
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find LLaVA decoder layers at model.model.layers")

    active: dict[str, Any] = {"outputs": {}, "iga": {}}
    image_mask, text_mask = image_and_text_masks(kwargs) if capture_iga else (None, None)
    handles = []
    for layer_index, layer in enumerate(layers):
        def save_layer(_module: Any, _inputs: Any, output: Any, *, index: int = layer_index) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                active["outputs"][index] = hidden.detach().to(device="cpu", dtype=torch.float16)

        handles.append(layer.register_forward_hook(save_layer))
        if capture_iga:
            attention_module = getattr(layer, "self_attn", None)
            if attention_module is None:
                raise RuntimeError(f"Decoder layer {layer_index} has no self_attn module")

            def save_attention(_module: Any, _inputs: Any, output: Any, *, index: int = layer_index) -> None:
                if not isinstance(output, tuple) or len(output) < 2 or not torch.is_tensor(output[1]):
                    return
                weights = output[1]
                if weights.ndim != 4:
                    return
                layer_scores = []
                for sample_index in range(weights.shape[0]):
                    query_mask = text_mask[sample_index].to(weights.device)
                    key_mask = image_mask[sample_index].to(weights.device)
                    text_to_image = weights[sample_index][:, query_mask, :][:, :, key_mask]
                    layer_scores.append(text_to_image.float().mean(dim=(0, 1)).cpu())
                active["iga"][index] = layer_scores
                # The model's output_attentions collection otherwise keeps a
                # full [batch, heads, sequence, sequence] matrix for every
                # decoder layer. IGA is already computed, so don't retain it.
                return (output[0], None, *output[2:])

            handles.append(attention_module.register_forward_hook(save_attention))

    try:
        if capture_iga:
            model.config.output_attentions = True
        device = next(model.parameters()).device
        with torch.inference_mode():
            result = adapter(
                inputs_embeds=inputs_embeds.to(device),
                attention_mask=kwargs["attention_mask"].to(device),
                labels=kwargs["labels"].to(device),
                use_cache=False,
                return_dict=True,
            )
        batch_count = inputs_embeds.shape[0]
        outputs = [
            [active["outputs"][layer_index][sample_index] for layer_index in range(len(layers))]
            for sample_index in range(batch_count)
        ]
        iga = None
        if capture_iga:
            missing = [index for index in range(len(layers)) if index not in active["iga"]]
            if missing:
                raise RuntimeError(
                    f"No post-softmax attention weights captured for layers {missing}; use eager attention."
                )
            iga = [
                [active["iga"][layer_index][sample_index] for layer_index in range(len(layers))]
                for sample_index in range(batch_count)
            ]
        logits = result.logits.detach().to(device="cpu", dtype=torch.float16)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, iga, logits


def layer_mse(
    base_outputs: list[list[torch.Tensor]],
    quant_outputs: list[list[torch.Tensor]],
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    iga_scores: list[list[torch.Tensor]],
    top_percent: float,
) -> tuple[list[float], list[float]]:
    full_errors, top_errors = [], []
    for layer_index, (base_y, quant_y) in enumerate(zip(base_outputs, quant_outputs)):
        if base_y.shape != quant_y.shape:
            raise ValueError(f"Layer {layer_index} output shapes differ: {base_y.shape} and {quant_y.shape}")
        difference = (base_y.float() - quant_y.float()).square().mean(dim=-1)
        valid = attention_mask.to(difference.device).bool()
        full_errors.append(float(difference[valid].mean()))
        image_positions = image_mask.to(difference.device).bool().nonzero(as_tuple=True)[0]
        scores = iga_scores[layer_index]
        token_count = max(1, math.ceil(image_positions.numel() * top_percent / 100.0))
        chosen_local = torch.topk(scores, k=token_count).indices
        chosen_positions = image_positions[chosen_local]
        top_errors.append(float(difference[chosen_positions].mean()))
    return top_errors, full_errors


def sample_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    shifted_logits = logits[:-1].float()
    shifted_labels = labels[1:].long()
    valid = shifted_labels != -100
    if not valid.any():
        raise ValueError("Sample has no assistant answer tokens for cross-entropy")
    token_losses = F.cross_entropy(
        shifted_logits,
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    return float(token_losses[valid].mean())


def sample_kl_divergence(
    base_logits: torch.Tensor,
    quant_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Average KL(base || quantized) over assistant answer tokens."""
    base_log_probs = F.log_softmax(base_logits[:-1].float(), dim=-1)
    quant_log_probs = F.log_softmax(quant_logits[:-1].float(), dim=-1)
    answer_labels = labels[1:].long()
    valid = answer_labels != -100
    if not valid.any():
        raise ValueError("Sample has no assistant answer tokens for KL divergence")
    token_kl = F.kl_div(
        quant_log_probs,
        base_log_probs.exp(),
        reduction="none",
    ).sum(dim=-1)
    return float(token_kl[valid].mean())


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    left = torch.tensor(x, dtype=torch.float64)
    right = torch.tensor(y, dtype=torch.float64)
    if left.numel() < 2 or left.std(unbiased=False) == 0 or right.std(unbiased=False) == 0:
        return None
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1])


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    try:
        if args.samples < 2:
            raise ValueError("--samples must be at least 2 to calculate correlations")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if not 0 < args.top_percent <= 100:
            raise ValueError("--top-percent must be in (0, 100]")
        run_started = time.perf_counter()

        quantized_dir = args.quantized_model.expanduser().resolve()
        metadata_path = quantized_dir / "qig_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"QIG metadata not found: {metadata_path}")
        model_config = read_yaml(args.model_config)
        dataset_paths = get_data_paths(args.dataset_config)
        base_model = args.base_model.expanduser().resolve() if args.base_model else repository_path(
            model_config.get("model_dir", "models/llava-v1.5-7b")
        )
        if not base_model.is_absolute():
            base_model = (REPOSITORY_ROOT / base_model).resolve()
        qig_source = Path(model_config.get("implementation_source_dir", "third_party/QIG")).expanduser()
        if not qig_source.is_absolute():
            qig_source = (REPOSITORY_ROOT / qig_source).resolve()
        runtime = load_qig_runtime(qig_source)
        log(
            f"Analysis config: base_model={base_model}, quantized_model={quantized_dir}, "
            f"device={args.device}, samples={args.samples}, batch_size={args.batch_size}, "
            f"top_percent={args.top_percent}, correlation_target={args.correlation_target}",
            level="extensive",
        )

        data_started = time.perf_counter()
        log("Loading ShareGPT4V COCO records with local images", flush=True)
        dataset = select_samples(args.dataset_config, args.samples, args.seed)
        log(f"Selected {len(dataset)} samples (seed={args.seed})", flush=True)
        if args.timing:
            log(f"[timing] Prepare dataset: {time.perf_counter() - data_started:.2f} s", flush=True)
        batches = [
            [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
            for start in range(0, len(dataset), args.batch_size)
        ]
        log(
            f"Prepared {len(batches)} batches; batch_sizes="
            f"{[len(batch) for batch in batches]}",
            level="extensive",
        )
        per_sample: list[dict[str, Any]] = []
        layer_count: int | None = None
        model_started = time.perf_counter()

        # Alternate checkpoint loads per batch to cap retained host/GPU memory.
        for batch_index, samples in enumerate(batches, start=1):
            batch_started = time.perf_counter()
            log(
                f"Batch {batch_index}/{len(batches)} sample_ids="
                f"{[str(sample.get('id', '')) for sample in samples]}",
                level="extensive",
            )
            base_load_started = time.perf_counter()
            log(f"Loading base model for batch {batch_index}/{len(batches)}", flush=True)
            base_lm, base_adapter = load_qig_llava_model(
                base_model,
                qig_source_dir=qig_source,
                device=args.device,
                attn_implementation=model_config.get("attn_implementation", "eager"),
            )
            if args.timing:
                log(f"[timing] Base model load: {time.perf_counter() - base_load_started:.2f} s", flush=True)
            base_adapter.model.config.output_attentions = True
            base_forward_started = time.perf_counter()
            inputs_embeds, batch_kwargs = prepare_batch(base_adapter, samples)
            base_outputs, iga_by_sample, _base_logits = capture_layer_outputs_and_iga(
                base_adapter,
                inputs_embeds,
                batch_kwargs,
                capture_iga=True,
            )
            if args.timing:
                log(f"[timing] Base preprocessing and forward: {time.perf_counter() - base_forward_started:.2f} s", flush=True)
            assert iga_by_sample is not None
            layer_count = len(base_outputs[0])
            log(
                f"Base capture: layers={layer_count}, inputs_embeds={tuple(inputs_embeds.shape)}, "
                f"attention_mask={tuple(batch_kwargs['attention_mask'].shape)}, "
                f"labels={tuple(batch_kwargs['labels'].shape)}, "
                f"vision_mask={tuple(batch_kwargs['vision_mask'].shape)}",
                level="extensive",
            )
            del base_adapter, base_lm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            quant_load_started = time.perf_counter()
            log(f"Loading quantized model for batch {batch_index}/{len(batches)}", flush=True)
            quant_lm = load_qig_quantized_model(
                quantized_dir,
                device=args.device,
                qig_source_dir=qig_source,
            )
            if args.timing:
                log(f"[timing] Quantized model load: {time.perf_counter() - quant_load_started:.2f} s", flush=True)
            quant_adapter = make_process_model(runtime, quant_lm)
            quant_forward_started = time.perf_counter()
            quant_outputs, _unused_iga, quant_logits = capture_layer_outputs_and_iga(
                quant_adapter,
                inputs_embeds,
                batch_kwargs,
                capture_iga=False,
            )
            if args.timing:
                log(f"[timing] Quantized forward: {time.perf_counter() - quant_forward_started:.2f} s", flush=True)
            if len(quant_outputs[0]) != layer_count:
                raise ValueError("Base and quantized models have different decoder layer counts")
            log(
                f"Quantized capture: layers={len(quant_outputs[0])}, "
                f"logits={tuple(quant_logits.shape)}, "
                f"base_layer0={tuple(base_outputs[0][0].shape)}, "
                f"quant_layer0={tuple(quant_outputs[0][0].shape)}",
                level="extensive",
            )

            image_mask = batch_kwargs["vision_mask"].bool()
            attention_mask = batch_kwargs["attention_mask"].bool()
            labels = batch_kwargs["labels"]
            for sample_index, sample in enumerate(samples):
                top_errors, full_errors = layer_mse(
                    base_outputs[sample_index],
                    quant_outputs[sample_index],
                    attention_mask[sample_index],
                    image_mask[sample_index],
                    iga_by_sample[sample_index],
                    args.top_percent,
                )
                ce = sample_cross_entropy(quant_logits[sample_index], labels[sample_index])
                base_ce = sample_cross_entropy(_base_logits[sample_index], labels[sample_index])
                ce_delta = ce - base_ce
                kl = sample_kl_divergence(
                    _base_logits[sample_index], quant_logits[sample_index], labels[sample_index]
                )
                per_sample.append(
                    {
                        "sample_index": len(per_sample),
                        "sample_id": str(sample.get("id", len(per_sample))),
                        "cross_entropy": ce,
                        "base_cross_entropy": base_ce,
                        "ce_delta": ce_delta,
                        "kl_base_to_quantized": kl,
                        "top_iga_mse": top_errors,
                        "full_mse": full_errors,
                    }
                )
                log(
                    f"Analyzed sample {len(per_sample)}/{len(dataset)}: "
                    f"KL={kl:.6g}, base_CE={base_ce:.6g}, quantized_CE={ce:.6g}, "
                    f"CE_delta={ce_delta:.6g}, top_IGA_MSE_mean="
                    f"{sum(top_errors) / len(top_errors):.6g}, full_MSE_mean="
                    f"{sum(full_errors) / len(full_errors):.6g}",
                    level="extensive",
                    flush=True,
                )
            log(f"Compared batch {batch_index}/{len(batches)}", flush=True)
            if args.timing:
                log(f"[timing] Batch total: {time.perf_counter() - batch_started:.2f} s", flush=True)
            del quant_adapter, quant_lm, inputs_embeds, batch_kwargs
            del base_outputs, quant_outputs, iga_by_sample, quant_logits, _base_logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if layer_count is None:
            raise RuntimeError("No model outputs were captured")
        target_key = "ce_delta" if args.correlation_target == "ce_delta" else "kl_base_to_quantized"
        targets = [row[target_key] for row in per_sample]
        layer_results = []
        for layer_index in range(layer_count):
            top_errors = [row["top_iga_mse"][layer_index] for row in per_sample]
            full_errors = [row["full_mse"][layer_index] for row in per_sample]
            corr_top = pearson_correlation(top_errors, targets)
            corr_full = pearson_correlation(full_errors, targets)
            layer_results.append(
                {
                    "layer": layer_index,
                    "corr_top_iga": corr_top,
                    "corr_full": corr_full,
                    "corr_difference": (
                        corr_top - corr_full
                        if corr_top is not None and corr_full is not None
                        else None
                    ),
                    "top_iga_mse_mean": sum(top_errors) / len(top_errors),
                    "full_mse_mean": sum(full_errors) / len(full_errors),
                }
            )

        output_dir = args.output_dir.expanduser() if args.output_dir else quantized_dir / "iga_error_correlation"
        if not output_dir.is_absolute():
            output_dir = (REPOSITORY_ROOT / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "base_model": str(base_model),
            "quantized_model": str(quantized_dir),
            "samples": len(per_sample),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "top_percent": args.top_percent,
            "correlation_target": args.correlation_target,
            "selection": "per sample and layer: highest IGA image tokens, IGA averaged over heads and text-query tokens",
            "output_mse_definition": "mean squared hidden-feature error at decoder block outputs; full uses all non-padding tokens, top uses selected image tokens",
            "correlation_target_definition": (
                "quantized model CE minus base model CE; each averaged over assistant answer tokens"
                if args.correlation_target == "ce_delta"
                else "KL divergence from base to quantized model, averaged over assistant answer tokens"
            ),
            "kl_definition": "per answer token: sum_vocab p_base * (log p_base - log p_quantized), averaged over assistant answer tokens",
            "layers": layer_results,
            "elapsed_seconds": time.perf_counter() - model_started,
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(["sample_index", "sample_id", "base_cross_entropy", "cross_entropy", "ce_delta", "kl_base_to_quantized", *[f"layer_{i}_top_iga_mse" for i in range(layer_count)], *[f"layer_{i}_full_mse" for i in range(layer_count)]])
            for row in per_sample:
                writer.writerow([row["sample_index"], row["sample_id"], row["base_cross_entropy"], row["cross_entropy"], row["ce_delta"], row["kl_base_to_quantized"], *row["top_iga_mse"], *row["full_mse"]])
        with (output_dir / "layer_correlations.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(layer_results[0]))
            writer.writeheader()
            writer.writerows(layer_results)

        log("Layer | corr_10% | corr_full | difference", flush=True)
        for row in layer_results:
            log(
                f"{row['layer']:>5} | {row['corr_top_iga']!s:>8} | "
                f"{row['corr_full']!s:>9} | {row['corr_difference']!s:>10}"
            )
        log(f"Saved result files to {output_dir}")
        if args.timing:
            log(f"[timing] Total: {time.perf_counter() - run_started:.2f} s", flush=True)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
