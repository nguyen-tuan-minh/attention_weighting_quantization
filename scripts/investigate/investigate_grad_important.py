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
def register_decoder_input_hooks(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture each decoder block's input activation without changing its input.

    Args:
        model: Loaded LLaVA causal language model with decoder blocks at
            ``model.model.layers``.
        active_sample: Mutable state whose ``layer_inputs`` key maps each
            decoder-layer index to that block's live input tensor
            [batch, sequence, hidden]. The mapping is cleared before each
            sample; the tensors remain attached to the current forward graph.

    Returns:
        Forward-pre-hook handles. The caller must remove every handle in a
        ``finally`` block.

    Hook contract:
        Read the first positional input to each decoder block, validate its
        [batch, sequence, hidden] shape, and store it by layer index. Return
        ``None`` so the original block input is left unchanged.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find decoder blocks at model.model.layers")

    # handles: list[torch.utils.hooks.RemovableHandle], hooks removed after all forwards.
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index, layer in enumerate(layers):
        def capture_block_input(
            _module: Any,
            inputs: tuple[Any, ...],
            *,
            index: int = layer_index,
        ) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"Decoder layer {index} received no tensor hidden-state input")
            # hidden_input: torch.Tensor, [batch, sequence, hidden], activation entering this decoder block.
            hidden_input = inputs[0]
            if hidden_input.ndim != 3:
                raise RuntimeError(
                    f"Decoder layer {index} input must be [batch, sequence, hidden], "
                    f"got {tuple(hidden_input.shape)}"
                )
            if not hidden_input.requires_grad:
                raise RuntimeError(
                    f"Decoder layer {index} input does not require gradients; "
                    "ensure the model forward runs with gradients enabled"
                )
            # active_sample["layer_inputs"][index]: torch.Tensor, live [batch, sequence, hidden] activation.
            active_sample["layer_inputs"][index] = hidden_input
            # Forward-pre-hook contract: None preserves the original positional inputs.

        handles.append(layer.register_forward_pre_hook(capture_block_input))

    if not handles:
        raise RuntimeError("No decoder blocks were found in the loaded model")
    return handles


# ============================================================================
# PHASE 4. FORWARD SAMPLES AND BACKPROPAGATE ASSISTANT TOKEN LOSSES
# ============================================================================
def score_assistant_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_mask: torch.Tensor,
    layer_inputs: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], list[float]]:
    """Sum squared decoder-input gradients separately for every answer-token CE.

    Args:
        logits: Model output logits [1, sequence, vocabulary] for one sample.
        labels: CPU integer token labels [sequence]; ``-100`` marks positions
            that are not assistant answer targets.
        image_mask: CPU boolean [sequence] mask selecting image tokens.
        layer_inputs: Layer index -> live decoder-block input tensor
            [1, sequence, hidden] captured by phase 3.

    Returns:
        ``scores`` maps layer index to a CPU float32 [image_tokens] vector. Each
        image-token score is the sum, over assistant answer tokens and hidden
        features, of squared gradients of that token's CE with respect to the
        corresponding decoder-block input. ``token_losses`` contains each
        scalar CE in answer-token order for logging.

    Assumptions:
        The forward batch has one sample, logits and labels have matching
        sequence length, and every captured block input participates in the
        loss graph. This performs one backward calculation per assistant token.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected one-sample logits [1, sequence, vocabulary], got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.numel() != logits.shape[1]:
        raise ValueError("Labels must be [sequence] and align with the model logits")
    if image_mask.ndim != 1 or image_mask.numel() != logits.shape[1]:
        raise ValueError("Image mask must be [sequence] and align with the model logits")
    if not layer_inputs:
        raise RuntimeError("Decoder input hooks captured no layer activations")

    # answer_positions: torch.Tensor, CPU long [answer_tokens], shifted label indices to score.
    answer_positions = labels[1:].ne(-100).nonzero(as_tuple=True)[0] + 1
    if answer_positions.numel() == 0:
        raise ValueError("Sample has no assistant answer tokens with valid CE targets")

    layer_indices = sorted(layer_inputs)
    captured_inputs = [layer_inputs[index] for index in layer_indices]
    for layer_index, activation in zip(layer_indices, captured_inputs):
        if activation.shape[0] != 1 or activation.shape[1] != logits.shape[1]:
            raise RuntimeError(
                f"Layer {layer_index} input shape {tuple(activation.shape)} does not align "
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

        # gradients: tuple[torch.Tensor, ...], one [1, sequence, hidden] gradient per decoder block input.
        gradients = torch.autograd.grad(
            token_loss,
            tuple(captured_inputs),
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
    display: bool = True,
    save_dir: Path | None = None,
) -> Path | None:
    """Display one image-plus-layer-grid figure and optionally save its PNG.

    Args:
        source_image: Sample image used by the QIG/LLaVA adapter.
        layer_scores: Layer index -> CPU float32 [image_tokens] gradient scores.
        sample_number: One-based index in the selected calibration subset.
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

    # Match QIG's square-pad preprocessing so the patch grid aligns with the displayed image.
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
    # image: np.ndarray, float32 [height, width, 3] in [0, 1], padded RGB display image.
    image = np.asarray(square_image).astype(np.float32) / 255.0
    image_height, image_width = image.shape[:2]

    # pooled_scores: torch.Tensor, CPU float32 [layers * image_tokens], values for one shared color scale.
    pooled_scores = torch.cat([scores.reshape(-1).float() for scores in layer_scores.values()])
    # positive_scores: torch.Tensor, CPU float32 [positive_values], strictly positive entries for LogNorm.
    positive_scores = pooled_scores[pooled_scores > 0]
    if positive_scores.numel() > 0:
        color_max = float(positive_scores.max())
        color_min = float(torch.quantile(positive_scores, 0.01))
        if color_min >= color_max:
            color_min = color_max * 1e-6
        color_norm = LogNorm(vmin=color_min, vmax=color_max, clip=True)
    else:
        # All scores are zero; linear normalization still produces a valid, legible figure.
        color_norm = Normalize(vmin=0.0, vmax=1.0)

    layer_indices = sorted(layer_scores)
    panel_count = len(layer_indices) + 1
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
    flat_axes[0].imshow(image)
    flat_axes[0].set_title("Input image")
    heatmap_view = None

    for axis, layer_index in zip(flat_axes[1:], layer_indices):
        # grid_scores: torch.Tensor, float32 [1, 1, grid_size, grid_size], this layer's image-token scores.
        grid_scores = layer_scores[layer_index].float().reshape(1, 1, grid_size, grid_size)
        # heatmap: np.ndarray, [height, width], bilinear-resized gradient-score grid.
        heatmap = F.interpolate(
            grid_scores,
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        axis.imshow(image)
        heatmap_view = axis.imshow(heatmap, cmap="inferno", norm=color_norm, alpha=0.58)
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
            label="Sum of squared assistant-token CE gradients",
        )
    figure.suptitle(
        f"Sample {sample_number}: decoder-block-input gradient importance",
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
            "cross-entropy gradients at every decoder block input."
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
        # model: torch.nn.Module, frozen LLaVA model whose block inputs are differentiated.
        # dataset: datasets.Dataset, selected examples processed one at a time.
        hooks_started = time.perf_counter() if args.timing else None
        # active_sample: dict[str, Any], live decoder input tensors for one current forward.
        # active_sample["layer_inputs"]: dict[int, torch.Tensor], layer -> live [batch, sequence, hidden] tensor.
        active_sample: dict[str, Any] = {"layer_inputs": {}}
        hook_handles = register_decoder_input_hooks(model, active_sample)
        log(f"Registered decoder-input hooks on {len(hook_handles)} layers")
        if hooks_started is not None:
            log(f"[timing] Register hooks: {time.perf_counter() - hooks_started:.2f} s")

        # =====================================================================
        # 4. FORWARD SAMPLES AND BACKPROPAGATE ASSISTANT TOKEN LOSSES
        # =====================================================================
        # Inputs from phase 3:
        # hook_handles: list[torch.utils.hooks.RemovableHandle], removed in finally after all samples.
        # active_sample["layer_inputs"]: dict[int, torch.Tensor], reset for each forward and kept live for token gradients.
        try:
            for sample_index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                # active_sample["layer_inputs"]: dict[int, torch.Tensor], reset before this sample's forward.
                active_sample["layer_inputs"] = {}
                # row: dict[str, Any], QIG adapter input with conversation, stable id, and local-image sentinel.
                row = {
                    "conversations": sample["conversations"],
                    "id": str(sample.get("id", sample_index)),
                    "image": "local",
                }
                image = sample["image"]
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
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
                        use_cache=False,
                        return_dict=True,
                    )
                    if not active_sample["layer_inputs"]:
                        raise RuntimeError("Decoder-input hooks captured no block activations")
                    expected_layers = len(model.model.layers)
                    missing_layers = sorted(
                        set(range(expected_layers)) - set(active_sample["layer_inputs"])
                    )
                    if missing_layers:
                        raise RuntimeError(f"Decoder-input hooks missed layers {missing_layers}")
                    # logits: torch.Tensor, [1, sequence, vocabulary], causal-LM output used for per-token CE.
                    logits = outputs.logits
                    forward_layer_inputs = active_sample["layer_inputs"]
                    # layer_scores: dict[int, torch.Tensor], CPU float32 [image_tokens] sum of per-answer-token grad squares.
                    layer_scores, token_losses = score_assistant_token_losses(
                        logits,
                        labels,
                        image_mask,
                        forward_layer_inputs,
                    )
                if forward_started is not None:
                    log(f"[timing] Sample {sample_index} forward and token gradients: {time.perf_counter() - forward_started:.2f} s")
                log(
                    f"[{sample_index}/{len(dataset)}] scored {len(token_losses)} assistant CE tokens; "
                    f"mean CE={sum(token_losses) / len(token_losses):.6g}",
                )

                # Clear references to the completed graph before drawing or forwarding the next sample.
                active_sample["layer_inputs"] = {}
                del outputs, logits, forward_layer_inputs, inputs_embeds
                plot_started = time.perf_counter() if args.timing else None
                output_path = show_gradient_heatmaps(
                    image,
                    layer_scores,
                    sample_index,
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
        # Phase 4 displayed one image-plus-layer-grid figure per selected sample.
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
