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


# ============================================================================
# CLI AND CONFIGURATION
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse model, dataset, sampling, metric, and logging options."""
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


# ============================================================================
# PHASE 1. PREPARE CALIBRATION DATASET
# ============================================================================
def select_samples(config_path: Path, count: int, seed: int | None) -> Dataset:
    """Load COCO records with local images and return a seeded sample subset.

    Args:
        config_path: Dataset YAML used by the shared ShareGPT4V loader.
        count: Number of records to return.
        seed: Optional shuffle seed; ``None`` lets the dataset choose ordering.

    Returns:
        A Dataset containing exactly ``count`` records with locally available images.
    """
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


# ============================================================================
# PHASE 2. LOAD MODEL(S)
# ============================================================================
def make_process_model(runtime: Any, lm: Any) -> Any:
    """Build QIG's LLaVA input adapter around a loaded language model.

    Args:
        runtime: Loaded QIG runtime exposing the ``llava`` process model.
        lm: Loaded model wrapper with model, tokenizer, and image processor.

    Returns:
        A QIG process adapter used for multimodal preprocessing and forward calls.
    """
    process_class = runtime.get_process_model("llava")
    return process_class(lm._model, lm._tokenizer, getattr(lm, "_image_processor", None))


# ============================================================================
# PHASES 3–4. REGISTER HOOKS AND FORWARD CALIBRATION SAMPLES
# ============================================================================
def prepare_batch(adapter: Any, samples: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preprocess samples and return CPU embeddings plus tensor input kwargs.

    Args:
        adapter: QIG LLaVA process adapter used for preprocessing/collation.
        samples: Dataset rows, each containing a conversation and local image.

    Returns:
        ``inputs_embeds`` with shape ``[batch, sequence, hidden]`` on CPU, and
        tensor kwargs (including ``attention_mask``, ``vision_mask``, and
        ``labels``), also detached on CPU for bookkeeping and later device moves.
    """
    # prepared_samples: list[Any], adapter-processed rows ready for collation.
    prepared_samples = []
    for index, sample in enumerate(samples):
        image = sample["image"]
        # The LLaVA pad preprocessor creates a canvas using an RGB background
        # color. Normalize grayscale/palette/RGBA inputs to RGB first.
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        # row: dict[str, Any], QIG adapter conversation/id/local-image input schema.
        row = {
            "conversations": sample["conversations"],
            "id": str(sample.get("id", index)),
            "image": "local",
        }
        prepared_samples.append(adapter.preprocess_data([image], row))
    batch = adapter.data_collator(prepared_samples)
    # prompt_inputs: dict[str, Any], model inputs including embeddings [batch, sequence, hidden].
    # prompt_kwargs: dict[str, Any], adapter metadata and tensors, commonly [batch, sequence].
    prompt_inputs, prompt_kwargs = adapter.generate_input(batch)
    # tensors: dict[str, torch.Tensor], adapter tensor fields detached to CPU with batch axes preserved.
    tensors = {
        key: value.detach().cpu()
        for key, value in prompt_kwargs.items()
        if torch.is_tensor(value)
    }
    # inputs_embeds: torch.Tensor, CPU [batch, sequence, hidden], detached and staged for both model passes.
    inputs_embeds = prompt_inputs["inputs_embeds"].detach().cpu()
    return inputs_embeds, tensors


def image_and_text_masks(kwargs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU image-key and post-image text-query masks, both [batch, sequence].

    Args:
        kwargs: Prepared adapter tensors containing attention and vision masks.

    Returns:
        ``image_mask`` and ``text_mask`` boolean tensors on CPU. Text queries are
        restricted to valid text positions after the final image token.
    """
    # Normalize adapter masks before boolean operations: some preprocessors
    # return attention_mask and vision_mask on different devices.
    # attention_mask: torch.Tensor, CPU bool [batch, sequence], valid-token mask.
    attention_mask = kwargs["attention_mask"].bool().detach().cpu()
    # image_mask: torch.Tensor | None, adapter vision mask before normalization.
    image_mask = kwargs.get("vision_mask")
    if image_mask is None:
        raise ValueError("QIG input adapter did not provide a vision_mask for image-token selection")
    # image_mask after normalization: torch.Tensor, CPU bool [batch, sequence], valid image tokens.
    image_mask = image_mask.bool().detach().cpu() & attention_mask
    # [batch, sequence] boolean masks partition every valid key position.
    # text_mask: torch.Tensor, CPU bool [batch, sequence], valid non-image positions before query filtering.
    text_mask = attention_mask & ~image_mask
    for sample_index in range(image_mask.shape[0]):
        # Image positions are an index vector [image_tokens] for this sample.
        # image_positions: torch.Tensor, long [image_tokens], image sequence indices for one sample.
        image_positions = image_mask[sample_index].nonzero(as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError(f"Sample {sample_index} has no image tokens")
        # As in investigate/investigate_attention.py, use text query tokens following the image.
        # text_mask[sample_index] after in-place filter: CPU bool [sequence], post-image text query positions.
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
    """Forward a batch and capture block outputs plus optional IGA scores.

    Args:
        adapter: QIG process adapter whose ``model`` contains LLaVA layers.
        inputs_embeds: CPU input embeddings with shape ``[batch, sequence, hidden]``.
        kwargs: CPU masks/labels consumed by the adapter.
        capture_iga: Whether to capture post-softmax attention and compute IGA.

    Returns:
        Per-sample, per-layer hidden outputs (each [sequence, hidden]); optional
        per-sample, per-layer image-token IGA vectors (each [image_tokens]); and
        logits with shape [batch, sequence, vocabulary]. Outputs are detached on CPU.
    """
    model = adapter.model
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find LLaVA decoder layers at model.model.layers")

    # active: dict[str, Any], hook captures reset for this model forward.
    # active["outputs"]: dict[int, torch.Tensor], layer -> CPU float16 [batch, sequence, hidden].
    # active["iga"]: dict[int, list[torch.Tensor]], layer -> per-sample CPU float32 [image_tokens].
    active: dict[str, Any] = {"outputs": {}, "iga": {}}
    # image_mask: torch.Tensor | None, CPU bool [batch, sequence], valid image tokens if IGA is enabled.
    # text_mask: torch.Tensor | None, CPU bool [batch, sequence], post-image text queries if IGA is enabled.
    image_mask, text_mask = image_and_text_masks(kwargs) if capture_iga else (None, None)
    # handles: list[torch.utils.hooks.RemovableHandle], hooks removed after the forward in finally.
    handles = []
    for layer_index, layer in enumerate(layers):
        def save_layer(_module: Any, _inputs: Any, output: Any, *, index: int = layer_index) -> None:
            # hidden: torch.Tensor, [batch, sequence, hidden], decoder block output.
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                # active["outputs"][index]: torch.Tensor, CPU float16 [batch, sequence, hidden], captured block output.
                active["outputs"][index] = hidden.detach().to(device="cpu", dtype=torch.float16)

        handles.append(layer.register_forward_hook(save_layer))
        if capture_iga:
            attention_module = getattr(layer, "self_attn", None)
            if attention_module is None:
                raise RuntimeError(f"Decoder layer {layer_index} has no self_attn module")

            def save_attention(_module: Any, _inputs: Any, output: Any, *, index: int = layer_index) -> None:
                if not isinstance(output, tuple) or len(output) < 2 or not torch.is_tensor(output[1]):
                    return
                # weights: torch.Tensor, [batch, heads, query_sequence, key_sequence], post-softmax attention.
                weights = output[1]
                if weights.ndim != 4:
                    return
                # layer_scores: list[torch.Tensor], CPU float32 [image_tokens] IGA vector per batch item.
                layer_scores = []
                for sample_index in range(weights.shape[0]):
                    # query_mask: torch.Tensor, bool [sequence] on attention device, text query positions.
                    query_mask = text_mask[sample_index].to(weights.device)
                    # key_mask: torch.Tensor, bool [sequence] on attention device, image key positions.
                    key_mask = image_mask[sample_index].to(weights.device)
                    # [heads, sequence, sequence] -> [heads, text_queries, image_keys].
                    # text_to_image: torch.Tensor, [heads, text_queries, image_tokens], selected attention values.
                    text_to_image = weights[sample_index][:, query_mask, :][:, :, key_mask]
                    # Reduce head/query axes -> [image_keys] IGA vector.
                    # layer_scores entry: torch.Tensor, CPU float32 [image_tokens], mean attention per image key.
                    layer_scores.append(text_to_image.float().mean(dim=(0, 1)).cpu())
                # active["iga"][index]: list[torch.Tensor], per-sample CPU [image_tokens] scores for this layer.
                active["iga"][index] = layer_scores
                # The model's output_attentions collection otherwise keeps a
                # full [batch, heads, sequence, sequence] matrix for every
                # decoder layer. IGA is already computed, so don't retain it.
                return (output[0], None, *output[2:])

            handles.append(attention_module.register_forward_hook(save_attention))

    try:
        if capture_iga:
            model.config.output_attentions = True
        # device: torch.device, model parameter device used for adapter inputs.
        device = next(model.parameters()).device
        with torch.inference_mode():
            # result: ModelOutput, logits [batch, sequence, vocabulary] plus model metadata.
            result = adapter(
                inputs_embeds=inputs_embeds.to(device),
                attention_mask=kwargs["attention_mask"].to(device),
                labels=kwargs["labels"].to(device),
                use_cache=False,
                return_dict=True,
            )
        batch_count = inputs_embeds.shape[0]
        # Split [batch, sequence, hidden] captured tensors into per-sample
        # [sequence, hidden] entries, grouped by sample and layer.
        # outputs: list[list[torch.Tensor]], CPU float16 [sequence, hidden] per sample and layer.
        outputs = [
            [active["outputs"][layer_index][sample_index] for layer_index in range(len(layers))]
            for sample_index in range(batch_count)
        ]
        # iga: list[list[torch.Tensor]] | None, optional CPU [image_tokens] vectors per sample/layer.
        iga = None
        if capture_iga:
            # missing: list[int], decoder layers for which hooks captured no attention weights.
            missing = [index for index in range(len(layers)) if index not in active["iga"]]
            if missing:
                raise RuntimeError(
                    f"No post-softmax attention weights captured for layers {missing}; use eager attention."
                )
            iga = [
                [active["iga"][layer_index][sample_index] for layer_index in range(len(layers))]
                for sample_index in range(batch_count)
            ]
        # logits: torch.Tensor, CPU float16 [batch, sequence, vocabulary].
        logits = result.logits.detach().to(device="cpu", dtype=torch.float16)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, iga, logits


# ============================================================================
# PHASE 5. ANALYSE CAPTURED DATA
# ============================================================================
def layer_mse(
    base_outputs: list[list[torch.Tensor]],
    quant_outputs: list[list[torch.Tensor]],
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    iga_scores: list[list[torch.Tensor]],
    top_percent: float,
) -> tuple[list[float], list[float]]:
    """Calculate top-IGA image-token and full-token MSE for every layer.

    Args:
        base_outputs: Per-layer reference outputs, each [sequence, hidden].
        quant_outputs: Per-layer quantized outputs with matching shapes.
        attention_mask: Valid-token mask [sequence].
        image_mask: Image-token mask [sequence].
        iga_scores: Per-layer IGA vectors, each [image_tokens].
        top_percent: Fraction of highest-IGA image tokens to include.

    Returns:
        Two per-layer lists: top-IGA image-token MSE and all-valid-token MSE.
    """
    # full_errors/top_errors: list[float], one scalar MSE per decoder layer.
    full_errors, top_errors = [], []
    for layer_index, (base_y, quant_y) in enumerate(zip(base_outputs, quant_outputs)):
        if base_y.shape != quant_y.shape:
            raise ValueError(f"Layer {layer_index} output shapes differ: {base_y.shape} and {quant_y.shape}")
        # [sequence, hidden] -> [sequence] feature-mean squared error.
        # difference: torch.Tensor, float [sequence], mean hidden-feature MSE per token.
        difference = (base_y.float() - quant_y.float()).square().mean(dim=-1)
        # valid: torch.Tensor, bool [sequence] on difference.device, non-padding positions.
        valid = attention_mask.to(difference.device).bool()
        full_errors.append(float(difference[valid].mean()))
        # image_positions: torch.Tensor, long [image_tokens] on difference.device.
        image_positions = image_mask.to(difference.device).bool().nonzero(as_tuple=True)[0]
        # scores: torch.Tensor, [image_tokens], per-image-token IGA ranking values.
        scores = iga_scores[layer_index]
        # token_count: int, number of highest-IGA image tokens selected for MSE.
        token_count = max(1, math.ceil(image_positions.numel() * top_percent / 100.0))
        # Chosen IGA ranks and resulting sequence positions are [top_token_count].
        # chosen_local: torch.Tensor, long [token_count], indices into scores/image_positions.
        chosen_local = torch.topk(scores, k=token_count).indices
        # chosen_positions: torch.Tensor, long [token_count], sequence positions of selected image tokens.
        chosen_positions = image_positions[chosen_local]
        top_errors.append(float(difference[chosen_positions].mean()))
    return top_errors, full_errors


def sample_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return quantized causal CE averaged over assistant answer tokens.

    Args:
        logits: Sample logits [sequence, vocabulary].
        labels: Aligned token labels [sequence], with ignored prompt positions -100.

    Returns:
        Mean next-token cross-entropy over non-ignored answer labels.
    """
    # Shift sequence positions so logits[:-1] predict labels[1:].
    # Shifted logits are [sequence - 1, vocabulary]; shifted labels are [sequence - 1].
    # shifted_logits: torch.Tensor, float [sequence - 1, vocabulary], predicts next-token labels.
    shifted_logits = logits[:-1].float()
    # shifted_labels: torch.Tensor, long [sequence - 1], targets aligned with preceding logits.
    shifted_labels = labels[1:].long()
    # valid: torch.Tensor, bool [sequence - 1], assistant answer positions only.
    valid = shifted_labels != -100
    if not valid.any():
        raise ValueError("Sample has no assistant answer tokens for cross-entropy")
    # Per-position loss vector is [sequence - 1]; valid selects answer positions.
    # token_losses: torch.Tensor, float [sequence - 1], unreduced causal CE by token.
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
    """Average KL(base || quantized) over assistant answer tokens.

    Args:
        base_logits: Reference logits [sequence, vocabulary].
        quant_logits: Quantized logits with matching shape.
        labels: Aligned token labels [sequence], with ignored prompt positions -100.

    Returns:
        Mean per-token KL divergence for non-ignored answer labels.
    """
    # Log-probability tensors retain [sequence - 1, vocabulary] after causal shift.
    # base_log_probs: torch.Tensor, float [sequence - 1, vocabulary], reference next-token log probabilities.
    base_log_probs = F.log_softmax(base_logits[:-1].float(), dim=-1)
    # quant_log_probs: torch.Tensor, float [sequence - 1, vocabulary], quantized next-token log probabilities.
    quant_log_probs = F.log_softmax(quant_logits[:-1].float(), dim=-1)
    # answer_labels: torch.Tensor, long [sequence - 1], shifted answer-token labels.
    answer_labels = labels[1:].long()
    # valid: torch.Tensor, bool [sequence - 1], positions included in answer-only KL.
    valid = answer_labels != -100
    if not valid.any():
        raise ValueError("Sample has no assistant answer tokens for KL divergence")
    # Sum vocabulary contributions to obtain one KL scalar per shifted token.
    # token_kl: torch.Tensor, float [sequence - 1], vocabulary-summed KL per position.
    token_kl = F.kl_div(
        quant_log_probs,
        base_log_probs.exp(),
        reduction="none",
    ).sum(dim=-1)
    return float(token_kl[valid].mean())


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Return Pearson correlation for equally sized per-sample metric vectors."""
    # Convert Python lists to matching float64 vectors of shape [samples].
    # left: torch.Tensor, float64 [samples], first per-sample metric vector.
    left = torch.tensor(x, dtype=torch.float64)
    # right: torch.Tensor, float64 [samples], second per-sample metric vector.
    right = torch.tensor(y, dtype=torch.float64)
    if left.numel() < 2 or left.std(unbiased=False) == 0 or right.std(unbiased=False) == 0:
        return None
    # Stack two [samples] metrics into [2, samples]; select their correlation.
    # correlation_matrix: torch.Tensor, float64 [2, 2], pairwise correlations of the two metric vectors.
    correlation_matrix = torch.corrcoef(torch.stack((left, right)))
    return float(correlation_matrix[0, 1])


def main() -> int:
    """Run dataset/model setup, capture both models, and correlate per-layer errors."""
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

        # =====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # =====================================================================
        # Inputs from setup:
        # args: argparse.Namespace, validated CLI options including sample/batch sizes.
        # dataset_config: dict[str, Any], YAML values used by the shared loader.
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
        # per_sample: list[dict[str, Any]], metric rows with sample identity,
        # CE/KL scalars, and per-layer top-IGA/full-token MSE lists.
        per_sample: list[dict[str, Any]] = []
        layer_count: int | None = None
        model_started = time.perf_counter()

        # =====================================================================
        # 2. LOAD MODELS (ALTERNATING CHECKPOINTS PER BATCH)
        # =====================================================================
        # Inputs from phase 1:
        # dataset: datasets.Dataset, selected rows with local images.
        # batches: list[list[dict[str, Any]]], Python row dictionaries grouped for forwarding.
        # per_sample: list[dict[str, Any]], initialized metric rows filled in phases 3–4.
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
            # 3–4. REGISTER BASE HOOKS AND FORWARD THE BATCH.
            # Hook setup, capture, and cleanup are grouped in the helper.
            # Inputs from phase 2:
            # base_adapter: Any, QIG LLaVA adapter that preprocesses and forwards this batch.
            # samples: list[dict[str, Any]], this batch's calibration rows.
            base_adapter.model.config.output_attentions = True
            base_forward_started = time.perf_counter()
            # inputs_embeds: torch.Tensor, CPU [batch, sequence, hidden], shared across model passes.
            # batch_kwargs: dict[str, torch.Tensor], detached CPU adapter tensors for this batch.
            inputs_embeds, batch_kwargs = prepare_batch(base_adapter, samples)
            # base_outputs: list[list[torch.Tensor]], CPU float16 [sequence, hidden] per sample/layer.
            # iga_by_sample: list[list[torch.Tensor]] | None, CPU [image_tokens] vectors per sample/layer.
            # _base_logits: torch.Tensor, CPU float16 [batch, sequence, vocabulary], reference logits.
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
            # =====================================================================
            # 3–4. REGISTER QUANTIZED HOOKS AND FORWARD THE SAME BATCH
            # =====================================================================
            # Inputs from the base pass:
            # inputs_embeds: torch.Tensor, CPU [batch, sequence, hidden], shared model input.
            # batch_kwargs: dict[str, torch.Tensor], CPU masks/labels reused unchanged.
            # quant_adapter: QIG LLaVA adapter wrapping the quantized model.
            # This pass captures block outputs; IGA is only needed from base.
            quant_forward_started = time.perf_counter()
            # quant_outputs: list[list[torch.Tensor]], CPU float16 [sequence, hidden] per sample/layer.
            # _unused_iga: None, IGA omitted for the quantized pass.
            # quant_logits: torch.Tensor, CPU float16 [batch, sequence, vocabulary], quantized logits.
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

            # image_mask: torch.Tensor, bool [batch, sequence], positions used for IGA-token MSE.
            image_mask = batch_kwargs["vision_mask"].bool()
            # attention_mask: torch.Tensor, bool [batch, sequence], valid positions used for full-token MSE.
            attention_mask = batch_kwargs["attention_mask"].bool()
            # labels: torch.Tensor, [batch, sequence], answer-token ids and -100 ignored positions.
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
                # Appended per_sample row: dict[str, Any], IDs, float CE/KL/MSE metrics, and per-layer float lists.
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

        # =====================================================================
        # 5. ANALYSE CAPTURED DATA
        # =====================================================================
        # Inputs from phases 3–4:
        # per_sample: list[dict[str, Any]], Python scalar metrics and per-layer float lists.
        # layer_count: int, decoder-layer count shared by base and quantized models.
        if layer_count is None:
            raise RuntimeError("No model outputs were captured")
        target_key = "ce_delta" if args.correlation_target == "ce_delta" else "kl_base_to_quantized"
        targets = [row[target_key] for row in per_sample]
        # layer_results: list[dict[str, Any]], one correlation/statistics row per decoder layer.
        layer_results = []
        for layer_index in range(layer_count):
            top_errors = [row["top_iga_mse"][layer_index] for row in per_sample]
            full_errors = [row["full_mse"][layer_index] for row in per_sample]
            corr_top = pearson_correlation(top_errors, targets)
            corr_full = pearson_correlation(full_errors, targets)
            # Appended layer_results row: dict[str, int | float | None], layer correlations and mean MSEs.
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
        # Summary schema: run/model metadata, sample-selection settings,
        # metric definitions, one result row per decoder layer, and elapsed time.
        # result: dict[str, Any], JSON summary with model/run metadata and layer_results.
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
