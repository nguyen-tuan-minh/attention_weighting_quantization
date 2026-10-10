"""Visualize image-token importance from assistant-token CE gradients."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.models import load_model  # noqa: E402
from investigation_logging import configure_logging, log  # noqa: E402


# ============================================================================
# PHASE 3. REGISTER HOOKS
# ============================================================================
def register_decoder_activation_hooks(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
    capture_point: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture selected decoder activations without changing model computation.

    Args:
        model: Loaded LLaVA causal language model with decoder blocks at
            ``model.model.layers``.
        active_sample: Mutable state whose ``layer_activations`` key maps each
            decoder-layer index to a live tensor [batch, sequence, hidden].
            The mapping is cleared before each sample; captured tensors remain
            attached to the current forward graph.
        capture_point: ``block_input`` for pre-normalization block inputs or
            ``after_layer_norm`` for the output of each block's input norm.

    Returns:
        Hook handles. The caller must remove every handle in a ``finally`` block.

    Hook contract:
        For ``block_input``, read the first positional input to each decoder
        block with a forward-pre-hook. For ``after_layer_norm``, read the
        output of that block's ``input_layernorm`` with a forward hook. Store
        the [batch, sequence, hidden] tensor by layer index; neither hook
        changes model values.
    """
    if capture_point not in {"block_input", "after_layer_norm"}:
        raise ValueError(f"Unsupported activation capture point: {capture_point}")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find decoder blocks at model.model.layers")

    # handles: list[torch.utils.hooks.RemovableHandle], hooks removed after all forwards.
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index, layer in enumerate(layers):
        def store_activation(activation: Any, *, index: int) -> None:
            if not torch.is_tensor(activation) or activation.ndim != 3:
                shape = tuple(activation.shape) if torch.is_tensor(activation) else type(activation).__name__
                raise RuntimeError(
                    f"Decoder layer {index} captured an invalid activation; expected "
                    f"[batch, sequence, hidden], got {shape}"
                )
            if not activation.requires_grad:
                raise RuntimeError(
                    f"Decoder layer {index} activation does not require gradients; "
                    "ensure the model forward runs with gradients enabled"
                )
            # active_sample["layer_activations"][index]: torch.Tensor, live [batch, sequence, hidden] selected activation.
            active_sample["layer_activations"][index] = activation

        if capture_point == "block_input":
            def capture_block_input(
                _module: Any,
                inputs: tuple[Any, ...],
                *,
                index: int = layer_index,
            ) -> None:
                if not active_sample.get("capture_enabled", True):
                    return
                if not inputs:
                    raise RuntimeError(f"Decoder layer {index} received no hidden-state input")
                store_activation(inputs[0], index=index)

            handles.append(layer.register_forward_pre_hook(capture_block_input))
        else:
            layer_norm = getattr(layer, "input_layernorm", None)
            if layer_norm is None:
                raise RuntimeError(f"Decoder layer {layer_index} has no input_layernorm module")

            def capture_after_layer_norm(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not active_sample.get("capture_enabled", True):
                    return
                store_activation(output, index=index)

            handles.append(layer_norm.register_forward_hook(capture_after_layer_norm))

    if not handles:
        raise RuntimeError("No decoder blocks were found in the loaded model")
    return handles


# ============================================================================
# PHASE 4. FORWARD SAMPLES AND BACKPROPAGATE ASSISTANT TOKEN LOSSES
# ============================================================================
def generate_assistant_reply(
    input_adapter: Any,
    image: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int,
) -> str:
    """Greedily decode one assistant response from a prompt-only multimodal row.

    Args:
        input_adapter: Loaded QIG LLaVA adapter, providing preprocessing,
            collation, multimodal prompt embeddings, model, and tokenizer.
        image: PIL image for the current calibration example.
        prompt_row: Adapter row containing conversation turns up to the latest
            user message, an id, and the local-image sentinel.
        max_new_tokens: Maximum number of generated assistant tokens.

    Returns:
        Decoded assistant response with tokenizer special tokens removed.

    Side effects:
        Runs repeated model forwards under inference mode, stopping on EOS or
        the token limit. It does not write files or retain generation activations.
    """
    # prepared_prompt: Any, preprocessed prompt-only multimodal row.
    prepared_prompt = input_adapter.preprocess_data([image], prompt_row)
    # prompt_batch: dict[str, Any], collated prompt consumed by generate_input.
    prompt_batch = input_adapter.data_collator([prepared_prompt])
    # prompt_inputs: dict[str, torch.Tensor], multimodal prompt embeddings [1, sequence, hidden].
    # prompt_kwargs: dict[str, torch.Tensor], expanded attention mask and all-ignored prompt labels.
    prompt_inputs, prompt_kwargs = input_adapter.generate_input(prompt_batch)
    # current_embeds: torch.Tensor, prompt [1, sequence, hidden] or one newest token [1, 1, hidden] with KV cache.
    current_embeds = prompt_inputs["inputs_embeds"]
    # current_attention_mask: torch.Tensor, device-local bool [1, cached_sequence + current_sequence].
    current_attention_mask = prompt_kwargs["attention_mask"]
    # generated_token_ids: list[torch.Tensor], device-local long [1, 1] ids emitted so far.
    generated_token_ids: list[torch.Tensor] = []
    # past_key_values: model cache for the prompt and generated prefix, or None when unsupported.
    past_key_values = None
    # eos_token_ids: set[int], configured end-of-sequence token ids that stop greedy decoding.
    eos_token_id = getattr(input_adapter.model.config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = input_adapter.tokenizer.eos_token_id
    eos_token_ids = set(eos_token_id) if isinstance(eos_token_id, (tuple, list)) else (
        {eos_token_id} if eos_token_id is not None else set()
    )
    # embedding_layer: torch.nn.Module, model token embedding table for extending prompt embeddings.
    embedding_layer = input_adapter.model.get_input_embeddings()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # outputs: ModelOutput, logits for the current sequence and an optional updated KV cache.
            outputs = input_adapter.model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            # next_token_id: torch.Tensor, device-local long [1, 1], greedy next-token choice.
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_token_ids.append(next_token_id)
            if int(next_token_id[0, 0]) in eos_token_ids:
                break

            # next_token_embed: torch.Tensor, [1, 1, hidden], embedding for the next cached forward.
            next_token_embed = embedding_layer(next_token_id)
            # next_attention: torch.Tensor, bool [1, 1], marks this generated token as valid.
            next_attention = torch.ones(
                (current_attention_mask.shape[0], 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            )
            # current_attention_mask: torch.Tensor, bool [1, total cached and current tokens].
            current_attention_mask = torch.cat((current_attention_mask, next_attention), dim=1)
            past_key_values = outputs.past_key_values
            if past_key_values is None:
                # Without a KV cache, append the new token embedding to the complete prefix for the next forward.
                current_embeds = torch.cat((current_embeds, next_token_embed), dim=1)
            else:
                # With a KV cache, only the newest token embedding is needed for the next forward.
                current_embeds = next_token_embed

    # generated_ids: torch.Tensor, [1, generated_tokens], all greedy continuation ids, including EOS if emitted.
    generated_ids = torch.cat(generated_token_ids, dim=1)
    # assistant_text: str, decoded generated continuation with tokenizer special tokens removed.
    assistant_text = input_adapter.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()
    del generated_ids, generated_token_ids, prompt_inputs, prompt_kwargs, prompt_batch, prepared_prompt
    return assistant_text


def score_assistant_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_mask: torch.Tensor,
    layer_activations: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], list[float]]:
    """Sum squared selected-activation gradients for each answer-token CE.

    Args:
        logits: Model output logits [1, sequence, vocabulary] for one sample.
        labels: CPU integer token labels [sequence]; ``-100`` marks positions
            that are not assistant answer targets.
        image_mask: CPU boolean [sequence] mask selecting image tokens.
        layer_activations: Layer index -> live selected activation tensor
            [1, sequence, hidden] captured by phase 3.

    Returns:
        ``scores`` maps layer index to a CPU float32 [image_tokens] vector. Each
        image-token score is the sum, over assistant answer tokens and hidden
        features, of squared gradients of that token's CE with respect to the
        corresponding selected decoder activation. ``token_losses`` contains each
        scalar CE in answer-token order for logging.

    Assumptions:
        The forward batch has one sample, logits and labels have matching
        sequence length, and every captured decoder activation participates in the
        loss graph. This performs one backward calculation per assistant token.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected one-sample logits [1, sequence, vocabulary], got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.numel() != logits.shape[1]:
        raise ValueError("Labels must be [sequence] and align with the model logits")
    if image_mask.ndim != 1 or image_mask.numel() != logits.shape[1]:
        raise ValueError("Image mask must be [sequence] and align with the model logits")
    if not layer_activations:
        raise RuntimeError("Activation hooks captured no layer activations")

    # answer_positions: torch.Tensor, CPU long [answer_tokens], shifted label indices to score.
    answer_positions = labels[1:].ne(-100).nonzero(as_tuple=True)[0] + 1
    if answer_positions.numel() == 0:
        raise ValueError("Sample has no assistant answer tokens with valid CE targets")

    layer_indices = sorted(layer_activations)
    captured_activations = [layer_activations[index] for index in layer_indices]
    for layer_index, activation in zip(layer_indices, captured_activations):
        if activation.shape[0] != 1 or activation.shape[1] != logits.shape[1]:
            raise RuntimeError(
                f"Layer {layer_index} activation shape {tuple(activation.shape)} does not align "
                "with the one-sample logits"
            )

    # scores: dict[int, torch.Tensor], layer -> CPU float32 zeros [image_tokens],
    # accumulated independently over answer-token losses for this sample.
    scores: dict[int, torch.Tensor] = {
        layer_index: torch.zeros(int(image_mask.sum()), dtype=torch.float32)
        for layer_index in layer_indices
    }
    # token_losses: list[float], detached scalar CE values in answer-token order.
    token_losses: list[float] = []
    answer_token_count = int(answer_positions.numel())

    for token_number, target_position_tensor in enumerate(answer_positions):
        target_position = int(target_position_tensor)
        # target: torch.Tensor, device-local long [1], one assistant token id.
        target = labels[target_position].to(device=logits.device, dtype=torch.long).reshape(1)
        # token_logits: torch.Tensor, float [1, vocabulary], preceding-position logits for this target.
        token_logits = logits[0, target_position - 1].float().unsqueeze(0)
        # token_loss: torch.Tensor, scalar, CE for this single assistant answer token.
        token_loss = F.cross_entropy(token_logits, target)
        token_losses.append(float(token_loss.detach().cpu()))

        # gradients: tuple[torch.Tensor, ...], one [1, sequence, hidden] gradient per selected layer activation.
        gradients = torch.autograd.grad(
            token_loss,
            tuple(captured_activations),
            retain_graph=token_number + 1 < answer_token_count,
            allow_unused=False,
        )
        for layer_index, gradient in zip(layer_indices, gradients):
            # image_gradient: torch.Tensor, float [image_tokens, hidden], this CE's image-token input gradients.
            image_gradient = gradient[0, image_mask.to(gradient.device), :].float()
            # scores[layer_index]: torch.Tensor, CPU float32 [image_tokens], sum of squared gradients so far.
            scores[layer_index] += image_gradient.square().sum(dim=-1).detach().cpu()

    return scores, token_losses


# ============================================================================
# PHASE 5. DISPLAY AND OPTIONALLY SAVE GRADIENT HEATMAPS
# ============================================================================
def show_gradient_heatmaps(
    source_image: Image.Image,
    layer_scores: dict[int, torch.Tensor],
    sample_number: int,
    *,
    heatmap_only: bool = False,
    scale: str = "log",
    score_normalization: str = "raw",
    display: bool = True,
    save_dir: Path | None = None,
) -> Path | None:
    """Display per-layer image-token heatmaps and optionally save the figure.

    Args:
        source_image: Sample image used by the QIG/LLaVA adapter.
        layer_scores: Layer index -> CPU float32 [image_tokens] gradient scores.
        sample_number: One-based index in the selected calibration subset.
        heatmap_only: Show the source image in a separate reference panel and
            patch-grid scores without image overlays when true.
        scale: Color normalization, either logarithmic or linear.
        score_normalization: ``raw`` preserves score magnitudes; ``percent``
            makes each image-token score a percentage of that layer's total.
        display: Whether to display the Matplotlib figure; true by default.
        save_dir: Optional directory enabling PNG output. ``None`` writes no file.

    Returns:
        Saved PNG path, or ``None`` when saving is disabled.

    Side effects:
        Optionally creates ``save_dir`` and writes a PNG, optionally displays a
        Matplotlib figure, and always closes that figure afterward.
    """
    if not layer_scores:
        raise ValueError("No per-layer gradient scores are available to plot")

    # image_token_count: int, number of image positions represented in each layer score vector.
    image_token_count = next(iter(layer_scores.values())).numel()
    # grid_size: int, side length for reshaping square image-token grids.
    grid_size = math.isqrt(image_token_count)
    if grid_size * grid_size != image_token_count:
        raise ValueError(
            f"Cannot reshape {image_token_count} image-token scores into a square patch grid"
        )
    if any(scores.numel() != image_token_count for scores in layer_scores.values()):
        raise ValueError("All layers must have the same number of image-token scores")

    # display_scores: dict[int, torch.Tensor], layer -> CPU float32 [image_tokens], raw or per-layer percentage values.
    if score_normalization == "percent":
        display_scores = {}
        for layer_index, scores in layer_scores.items():
            # layer_total: torch.Tensor, CPU float32 scalar, sum of this layer's image-token scores.
            layer_total = scores.sum()
            if layer_total > 0:
                # display_scores[layer_index]: torch.Tensor, CPU float32 [image_tokens], each token's percent of layer total.
                display_scores[layer_index] = scores.float() / layer_total * 100.0
            else:
                # All-zero scores have no defined share; represent every token as 0 percent.
                display_scores[layer_index] = torch.zeros_like(scores, dtype=torch.float32)
    elif score_normalization == "raw":
        # display_scores: dict[int, torch.Tensor], raw CPU scores retained for each layer.
        display_scores = layer_scores
    else:
        raise ValueError(f"Unknown score normalization: {score_normalization}")

    # image: np.ndarray, float32 [square_side, square_side, 3] in [0, 1], square-padded RGB sample.
    source_image = source_image.convert("RGB")
    image_width, image_height = source_image.size
    square_side = max(image_width, image_height)
    # background: tuple[int, int, int], CLIP mean RGB used to fill the square-padded canvas.
    background = tuple(round(value * 255) for value in (0.48145466, 0.4578275, 0.40821073))
    square_image = Image.new("RGB", (square_side, square_side), background)
    square_image.paste(
        source_image,
        ((square_side - image_width) // 2, (square_side - image_height) // 2),
    )
    image = np.asarray(square_image).astype(np.float32) / 255.0

    # pooled_scores: torch.Tensor, CPU float32 [layers * image_tokens], values for one shared color scale.
    pooled_scores = torch.cat([scores.reshape(-1).float() for scores in display_scores.values()])
    # positive_scores: torch.Tensor, CPU float32 [positive_values], strictly positive entries for LogNorm.
    positive_scores = pooled_scores[pooled_scores > 0]
    if scale == "linear":
        # color_min/color_max: float, shared linear scale limits across all layers in this sample.
        color_min = float(pooled_scores.min())
        color_max = float(pooled_scores.max())
        if color_min >= color_max:
            color_max = color_min + 1.0
        color_norm = Normalize(vmin=color_min, vmax=color_max, clip=True)
    elif positive_scores.numel() > 0:
        color_max = float(positive_scores.max())
        color_min = float(torch.quantile(positive_scores, 0.01))
        if color_min >= color_max:
            color_min = color_max * 1e-6
        color_norm = LogNorm(vmin=color_min, vmax=color_max, clip=True)
    else:
        # All scores are zero; linear normalization still produces a valid, legible figure.
        color_norm = Normalize(vmin=0.0, vmax=1.0)

    layer_indices = sorted(display_scores)
    # panel_count: int, layer heatmap panels plus an image-reference panel in heatmap-only mode.
    panel_count = len(layer_indices) + int(heatmap_only)
    columns = 6
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3.5, rows * 3.2),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    heatmap_view = None
    # first_heatmap_axis: int, first panel index reserved for per-layer heatmaps.
    first_heatmap_axis = 1 if heatmap_only else 0
    if heatmap_only:
        flat_axes[0].imshow(image)
        flat_axes[0].set_title("Reference image")

    for axis, layer_index in zip(flat_axes[first_heatmap_axis:], layer_indices):
        # grid_scores: torch.Tensor, float32 [1, 1, grid_size, grid_size], this layer's image-token scores.
        grid_scores = display_scores[layer_index].float().reshape(1, 1, grid_size, grid_size)
        # heatmap: torch.Tensor, CPU float32 [grid_size, grid_size], image-token scores in patch-grid layout.
        heatmap = grid_scores[0, 0]
        if not heatmap_only:
            axis.imshow(image)
            # display_heatmap: np.ndarray, float32 [square_side, square_side], resized scores over source image.
            display_heatmap = F.interpolate(
                grid_scores,
                size=(square_side, square_side),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
        else:
            # display_heatmap: np.ndarray, float32 [grid_size, grid_size], raw image-token score grid.
            display_heatmap = heatmap.numpy()
        heatmap_view = axis.imshow(
            display_heatmap,
            cmap="inferno",
            norm=color_norm,
            alpha=0.58 if not heatmap_only else 1.0,
        )
        axis.set_title(f"Layer {layer_index}")

    for axis in flat_axes[panel_count:]:
        axis.axis("off")
    for axis in flat_axes[:panel_count]:
        axis.axis("off")
    if heatmap_view is not None:
        figure.colorbar(
            heatmap_view,
            ax=flat_axes[:panel_count].tolist(),
            fraction=0.015,
            pad=0.01,
            label=(
                "Percent of this layer's image-token gradient score"
                if score_normalization == "percent"
                else "Sum of squared assistant-token CE gradients"
            ),
        )
    figure.suptitle(
        f"Sample {sample_number}: selected decoder-activation gradient importance",
        fontsize=14,
    )

    output_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"sample_{sample_number:04d}_grad_important.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        log(f"Saved gradient heatmaps to {output_path}")
    if display:
        plt.show()
    plt.close(figure)
    return output_path


# ============================================================================
# CLI AND CONFIGURATION
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse shared config, sample selection, display/save, and logging options."""
    parser = argparse.ArgumentParser(
        description=(
            "Display image-token importance maps from separate assistant answer-token "
            "cross-entropy gradients at selected decoder activations."
        )
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "dataset.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "model.yaml",
    )
    parser.add_argument("--samples", type=int, default=None, help="Override configured sample count.")
    parser.add_argument("--seed", type=int, default=None, help="Override configured sample seed.")
    parser.add_argument(
        "--capture-point",
        choices=("block_input", "after_layer_norm"),
        default="block_input",
        help="Activation to differentiate: pre-norm block input (default) or input-layer-norm output.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("example", "generate"),
        default="example",
        help="Score the dataset's assistant answer (default) or a greedily generated reply.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated assistant tokens when --answer-mode generate is selected.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for per-sample PNGs; omitted means no files are saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open figures; useful when saving with --save-dir in a headless run.",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Show the source image as a reference panel and layer heatmaps without overlays.",
    )
    parser.add_argument(
        "--scale",
        choices=("log", "linear"),
        default="log",
        help="Heatmap color scale: logarithmic (default) or linear.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=("raw", "percent"),
        default="raw",
        help="Show raw scores or each image token's percentage of its layer's total score.",
    )
    parser.add_argument(
        "--log-level",
        choices=("none", "normal", "extensive"),
        default="normal",
        help="Output detail: none, concise progress, or detailed diagnostics and timings.",
    )
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress Python and dependency warnings.")
    parser.add_argument("--timing", action="store_true", help="Print elapsed time for setup and each sample.")
    return parser.parse_args()


def main() -> int:
    """Run dataset setup, model loading, hooks, per-token backprop, and display."""
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    try:
        # Read options and validate settings before loading large resources.
        dataset_config = read_yaml(args.dataset_config)
        model_config = read_yaml(args.model_config)
        sample_count = (
            args.samples
            if args.samples is not None
            else dataset_config.get("calibration_samples", 2)
        )
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("calibration_samples must be a positive integer")
        if not isinstance(args.max_new_tokens, int) or args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be a positive integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("calibration_seed must be an integer or null")
        if args.save_dir is not None:
            save_dir = args.save_dir.expanduser()
            if not save_dir.is_absolute():
                save_dir = (REPOSITORY_ROOT / save_dir).resolve()
        else:
            save_dir = None

        total_started = time.perf_counter() if args.timing else None

        # =====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # =====================================================================
        # Inputs from setup:
        # args: argparse.Namespace, parsed sample/config/display options.
        # dataset_config: dict[str, Any], dataset YAML values.
        dataset_started = time.perf_counter() if args.timing else None
        dataset: Dataset = load_sharegpt4v_dataset(
            source="coco",
            config_path=args.dataset_config,
            download_images=False,
            existing_images_only=True,
        )
        if sample_count > len(dataset):
            raise ValueError(
                f"Requested {sample_count} samples, but only {len(dataset)} COCO records "
                "with local images are available"
            )
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))
        log(f"Selected {len(dataset):,} COCO samples with local images")
        log(
            f"Dataset config={args.dataset_config}, requested={sample_count}, seed={seed}",
            level="extensive",
        )
        if dataset_started is not None:
            log(f"[timing] Prepare dataset: {time.perf_counter() - dataset_started:.2f} s")

        # =====================================================================
        # 2. LOAD MODEL
        # =====================================================================
        # Inputs from phase 1:
        # dataset: datasets.Dataset, selected COCO rows with local images.
        # model_config: dict[str, Any], model YAML values loaded during setup.
        model_started = time.perf_counter() if args.timing else None
        model_id = model_config["model_id"]
        model_dir = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        has_weights = any(model_dir.glob("pytorch_model*.bin")) or any(
            model_dir.glob("model*.safetensors")
        )
        if not (model_dir / "config.json").is_file() or not has_weights:
            model_dir.mkdir(parents=True, exist_ok=True)
            log(f"Downloading LLaVA checkpoint {model_id} to {model_dir}")
            snapshot_download(repo_id=model_id, local_dir=str(model_dir))

        model, input_adapter = load_model(
            model_dir,
            model_config,
            repository_root=REPOSITORY_ROOT,
        )
        model.eval()
        # Freeze model weights while still allowing gradients through activations that require gradients.
        model.requires_grad_(False)
        model_parameter = next(model.parameters())
        log(
            f"Loaded model_id={model_id}, model_dir={model_dir}, "
            f"device={model_parameter.device}, dtype={model_parameter.dtype}, "
            f"attn_implementation={model_config.get('attn_implementation', 'eager')}",
            level="extensive",
        )
        if model_started is not None:
            log(f"[timing] Load model: {time.perf_counter() - model_started:.2f} s")

        # =====================================================================
        # 3. REGISTER HOOKS
        # =====================================================================
        # Inputs from phases 1–2:
        # model: torch.nn.Module, frozen LLaVA model whose selected decoder activations are differentiated.
        # dataset: datasets.Dataset, selected examples processed one at a time.
        hooks_started = time.perf_counter() if args.timing else None
        # active_sample: dict[str, Any], hook state for the current sample and selected forward.
        # active_sample["layer_activations"]: dict[int, torch.Tensor], layer -> live [batch, sequence, hidden] activation.
        # active_sample["capture_enabled"]: bool, false during inference-only answer generation, true for scored forward.
        active_sample: dict[str, Any] = {"layer_activations": {}, "capture_enabled": True}
        hook_handles = register_decoder_activation_hooks(model, active_sample, args.capture_point)
        log(f"Registered {args.capture_point} hooks on {len(hook_handles)} layers")
        if hooks_started is not None:
            log(f"[timing] Register hooks: {time.perf_counter() - hooks_started:.2f} s")

        # =====================================================================
        # 4. FORWARD SAMPLES AND BACKPROPAGATE ASSISTANT TOKEN LOSSES
        # =====================================================================
        # Inputs from phase 3:
        # hook_handles: list[torch.utils.hooks.RemovableHandle], removed in finally after all samples.
        # active_sample["layer_activations"]: dict[int, torch.Tensor], reset per forward and retained for token gradients.
        try:
            for sample_index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                # active_sample["layer_activations"]: dict[int, torch.Tensor], reset before this sample's scored forward.
                active_sample["layer_activations"] = {}
                # source_conversations: list[dict[str, str]], original user/assistant turns from the dataset.
                source_conversations = sample["conversations"]
                # row: dict[str, Any], QIG adapter input with conversation, stable id, and local-image sentinel.
                row = {
                    "conversations": source_conversations,
                    "id": str(sample.get("id", sample_index)),
                    "image": "local",
                }
                image = sample["image"]
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                if args.answer_mode == "generate":
                    # user_turn_indices: list[int], positions of user turns available as generation context.
                    user_turn_indices = [
                        index
                        for index, turn in enumerate(source_conversations)
                        if turn.get("from") in {"human", "user"}
                    ]
                    if not user_turn_indices:
                        raise ValueError(f"Sample {row['id']} has no user turn to generate from")
                    # prompt_conversations: list[dict[str, str]], turns through the latest user message, without its answer.
                    prompt_conversations = source_conversations[: user_turn_indices[-1] + 1]
                    # prompt_row: dict[str, Any], user-context row consumed only for assistant generation.
                    prompt_row = {**row, "conversations": prompt_conversations}
                    generation_started = time.perf_counter() if args.timing else None
                    # active_sample["capture_enabled"]: bool, suspend gradient hooks during no-grad generation.
                    active_sample["capture_enabled"] = False
                    try:
                        assistant_text = generate_assistant_reply(
                            input_adapter,
                            image,
                            prompt_row,
                            args.max_new_tokens,
                        )
                    finally:
                        # active_sample["capture_enabled"]: bool, re-enable hooks for the answer-scoring forward.
                        active_sample["capture_enabled"] = True
                    if not assistant_text:
                        raise ValueError(f"Sample {row['id']} produced an empty assistant response")
                    log(
                        f"[{sample_index}/{len(dataset)}] GENERATED ASSISTANT: {assistant_text}",
                        level="normal",
                    )
                    if generation_started is not None:
                        log(
                            f"[timing] Sample {sample_index} generation: "
                            f"{time.perf_counter() - generation_started:.2f} s"
                        )
                    # row["conversations"]: list[dict[str, str]], prompt turns plus generated assistant answer for scoring.
                    row["conversations"] = prompt_conversations + [
                        {"from": "gpt", "value": assistant_text}
                    ]
                # prepared: Any, one adapter-preprocessed multimodal row.
                prepared = input_adapter.preprocess_data([image], row)
                # batch: dict[str, Any], collated single-sample input to generate_input.
                batch = input_adapter.data_collator([prepared])
                # prompt_inputs: dict[str, Any], includes inputs_embeds [1, sequence, hidden].
                # prompt_kwargs: dict[str, Any], adapter attention/vision masks and labels.
                prompt_inputs, prompt_kwargs = input_adapter.generate_input(batch)

                # attention_mask: torch.Tensor, CPU bool [sequence], valid token positions.
                attention_mask = prompt_kwargs["attention_mask"][0].bool().detach().cpu()
                # image_mask: torch.Tensor, CPU bool [sequence], image-token positions restricted to valid tokens.
                image_mask = prompt_kwargs["vision_mask"][0].bool().detach().cpu() & attention_mask
                # labels: torch.Tensor, CPU long [sequence], assistant target ids and -100 ignored positions.
                labels = prompt_kwargs["labels"][0].detach().to(device="cpu", dtype=torch.long)
                answer_token_count = int(labels[1:].ne(-100).sum())
                if not image_mask.any():
                    raise ValueError(f"Sample {row['id']} has no image tokens")
                if answer_token_count == 0:
                    raise ValueError(f"Sample {row['id']} has no assistant answer tokens")

                # inputs_embeds: torch.Tensor, gradient-enabled [1, sequence, hidden] leaf on its adapter device.
                # Detach prevents unnecessary gradient tracking through embedding/vision-tower computation.
                inputs_embeds = prompt_inputs["inputs_embeds"].detach().requires_grad_(True)
                log(
                    f"Sample {sample_index}/{len(dataset)} id={row['id']}: "
                    f"embeds={tuple(inputs_embeds.shape)}, image_tokens={int(image_mask.sum())}, "
                    f"assistant_tokens={answer_token_count}",
                    level="extensive",
                )

                forward_started = time.perf_counter() if args.timing else None
                with torch.enable_grad():
                    # outputs: ModelOutput, logits [1, sequence, vocabulary] and model metadata.
                    outputs = input_adapter(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prompt_kwargs["attention_mask"],
                        # QIG's LLaVA adapter requires labels and uses them only to populate its output loss;
                        # assistant-token gradients below still use the individual logits-derived CEs.
                        labels=prompt_kwargs["labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                    if not active_sample["layer_activations"]:
                        raise RuntimeError("Activation hooks captured no decoder activations")
                    expected_layers = len(model.model.layers)
                    missing_layers = sorted(
                        set(range(expected_layers)) - set(active_sample["layer_activations"])
                    )
                    if missing_layers:
                        raise RuntimeError(f"Activation hooks missed layers {missing_layers}")
                    # logits: torch.Tensor, [1, sequence, vocabulary], causal-LM output used for per-token CE.
                    logits = outputs.logits
                    # forward_layer_activations: dict[int, torch.Tensor], layer -> live selected [1, sequence, hidden] activation.
                    forward_layer_activations = active_sample["layer_activations"]
                    # layer_scores: dict[int, torch.Tensor], CPU float32 [image_tokens] sum of per-answer-token grad squares.
                    layer_scores, token_losses = score_assistant_token_losses(
                        logits,
                        labels,
                        image_mask,
                        forward_layer_activations,
                    )
                if forward_started is not None:
                    log(f"[timing] Sample {sample_index} forward and token gradients: {time.perf_counter() - forward_started:.2f} s")
                log(
                    f"[{sample_index}/{len(dataset)}] scored {len(token_losses)} assistant CE tokens; "
                    f"mean CE={sum(token_losses) / len(token_losses):.6g}",
                )

                # Clear references to the completed graph before drawing or forwarding the next sample.
                active_sample["layer_activations"] = {}
                del outputs, logits, forward_layer_activations, inputs_embeds
                plot_started = time.perf_counter() if args.timing else None
                output_path = show_gradient_heatmaps(
                    image,
                    layer_scores,
                    sample_index,
                    heatmap_only=args.heatmap_only,
                    scale=args.scale,
                    score_normalization=args.score_normalization,
                    display=not args.no_display,
                    save_dir=save_dir,
                )
                if output_path is None:
                    log(f"Displayed gradient heatmaps for sample {sample_index}")
                if plot_started is not None:
                    log(f"[timing] Sample {sample_index} plot: {time.perf_counter() - plot_started:.2f} s")
                if sample_started is not None:
                    log(f"[timing] Sample {sample_index} total: {time.perf_counter() - sample_started:.2f} s")
                del layer_scores, token_losses, prompt_inputs, prompt_kwargs, batch, prepared, image, sample
        finally:
            for handle in hook_handles:
                handle.remove()

        # =====================================================================
        # 5. DISPLAY AND OPTIONALLY SAVE RESULTS
        # =====================================================================
        # Phase 4 displayed one per-layer heatmap grid per sample; --heatmap-only omits source-image backgrounds.
        # PNG files are written only when ``--save-dir`` was supplied.
        log(f"Gradient-importance analysis complete for {len(dataset):,} samples.")
        if save_dir is not None:
            log(f"Saved per-sample figures under: {save_dir}")
        if total_started is not None:
            log(f"[timing] Total: {time.perf_counter() - total_started:.2f} s")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
