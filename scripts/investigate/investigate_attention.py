"""Run calibration examples and analyse or save language attention weights."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
from datasets import Dataset
from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import (  # noqa: E402
    get_data_paths,
    load_sharegpt4v_dataset,
)
from attention_quantization.models import load_model  # noqa: E402
from investigation_logging import configure_logging, log  # noqa: E402


# ============================================================================
# PHASE 3. REGISTER HOOKS
# ============================================================================
def register_attention_hooks(
    model: torch.nn.Module,
    mode: str,
    active_sample: dict[str, Any],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register language-attention hooks for IGA maps or saved matrices.

    Args:
        model: Loaded LLaVA model containing decoder self-attention modules.
        mode: ``online`` reduces attention to per-image-token IGA scores;
            ``save`` writes raw matrices to the active sample directory.
        active_sample: Shared state schema:
            ``directory`` is an optional output path; ``captured`` is the
            current sample's matrix count; ``text_mask`` and ``image_mask`` are
            boolean [sequence] masks; ``iga`` maps layer names to CPU
            [image_tokens] scores. Reset sample-scoped values before each
            forward.

    Returns:
        Hook handles that the caller must remove in a ``finally`` block.

    Hook contract:
        Read post-softmax weights [batch, heads, query, key]. In online mode,
        select text queries and image keys, then average over heads and queries
        to produce one [image_tokens] vector. In save mode, write the raw matrix
        as CPU float16. The attention output is left unchanged.
    """
    handles = []
    for name, module in model.named_modules():
        if not name.endswith(".self_attn") or "vision_tower" in name:
            continue

        layer_name = name.replace(".", "_")

        def save_attention(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            key: str = layer_name,
        ) -> None:
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attention_weights = output[1]
            if not torch.is_tensor(attention_weights):
                return

            if mode == "save":
                output_dir = active_sample["directory"]
                if output_dir is None:
                    return
                # Saved attention shape is [batch, heads, sequence, sequence].
                torch.save(
                    attention_weights.detach().to(device="cpu", dtype=torch.float16),
                    output_dir / f"{key}.pt",
                )
            else:
                # For each layer, average post-softmax attention over text
                # query positions and heads to get one score per image token.
                text_mask = active_sample["text_mask"].to(attention_weights.device)
                image_mask = active_sample["image_mask"].to(attention_weights.device)
                # Shape after selection: heads x text queries x image keys.
                # Select one batch item then query/key masks:
                # [heads, sequence, sequence] -> [heads, text_queries, image_keys].
                text_to_image = attention_weights[0][:, text_mask, :][:, :, image_mask]
                # Average heads and text queries -> one score per image key.
                active_sample["iga"][key] = text_to_image.float().mean(dim=(0, 1)).cpu()
            active_sample["captured"] += 1

        handles.append(module.register_forward_hook(save_attention))

    if not handles:
        raise RuntimeError("No text self-attention modules were found in the loaded model")
    return handles


# ============================================================================
# PHASE 5. ANALYSE CAPTURED ATTENTION
# ============================================================================
def show_iga_heatmap(
    source_image: Image.Image,
    layer_iga_scores: dict[str, torch.Tensor],
    sample_number: int,
    heatmap_only: bool = False,
    attention_percent: bool = False,
    top_iga_percent: float | None = None,
) -> None:
    """Display one image-key attention/IGA map per decoder layer.

    Args:
        source_image: Original sample image used as the overlay background.
        layer_iga_scores: Mapping from layer name to CPU [image_tokens] scores.
        sample_number: One-based sample number for the figure title.
        heatmap_only: Draw maps without the image overlay when true.
        attention_percent: Scale softmax attention weights by 100 for display.
        top_iga_percent: Optional fraction in (0, 1] to retain by IGA score
            separately per layer and display as a binary map.

    Side effects:
        Displays a Matplotlib figure and closes it after the window is dismissed.
    """
    if not layer_iga_scores:
        raise ValueError("No per-layer IGA scores are available to plot")

    token_count = next(iter(layer_iga_scores.values())).numel()
    grid_size = math.isqrt(token_count)
    if grid_size * grid_size != token_count:
        raise ValueError(
            f"Cannot reshape {token_count} image-token scores into a square patch grid"
        )

    # QIG's classic LLaVA preprocessing pads to square before resizing.
    source_image = source_image.convert("RGB")
    width, height = source_image.size
    side = max(width, height)
    background = tuple(round(value * 255) for value in (0.48145466, 0.4578275, 0.40821073))
    square_image = Image.new("RGB", (side, side), background)
    square_image.paste(source_image, ((side - width) // 2, (side - height) // 2))
    # Convert the padded RGB image to a float [height, width, 3] array in [0, 1].
    image = np.asarray(square_image).astype(np.float32) / 255.0

    image_height, image_width = image.shape[:2]
    # Use a shared logarithmic scale so small IGA differences remain visible
    # while keeping layer magnitudes comparable within this sample.
    binary_mode = top_iga_percent is not None
    if binary_mode:
        layer_scores = []
        for scores in layer_iga_scores.values():
            # Rank each layer's image tokens by its IGA score and retain the
            # requested fraction of those scores.
            retain_count = max(1, math.ceil(scores.numel() * top_iga_percent))
            # Flatten [image_tokens] before ranking; indices has [retain_count].
            selected = torch.topk(scores.flatten(), k=retain_count).indices
            binary_scores = torch.zeros_like(scores).flatten()
            binary_scores[selected] = 1.0
            layer_scores.append(binary_scores.reshape_as(scores))
        color_map = ListedColormap(["#202020", "#FFD400"])
        color_norm = BoundaryNorm([-0.5, 0.5, 1.5], color_map.N)
    else:
        scale = 100.0 if attention_percent else 1.0
        layer_scores = [scores * scale for scores in layer_iga_scores.values()]
        # Pool layer maps as one [layers * image_tokens] vector for shared scaling.
        positive_scores = torch.cat([scores.reshape(-1) for scores in layer_scores])
        positive_scores = positive_scores[positive_scores > 0]
        if positive_scores.numel() == 0:
            raise ValueError("IGA scores are all zero; cannot draw a logarithmic heatmap")
        color_min = float(torch.quantile(positive_scores, 0.01))
        color_max = float(positive_scores.max())
        if color_min >= color_max:
            color_min = color_max * 1e-6
        color_norm = LogNorm(vmin=color_min, vmax=color_max, clip=True)
        color_map = "inferno"

    # Display the image and all layer overlays together in one figure per sample.
    panel_count = len(layer_iga_scores) + 1
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
    flat_axes[0].set_title("Processor image")

    heatmap_view = None
    for axis, (layer_name, scores) in zip(flat_axes[1:], zip(layer_iga_scores, layer_scores)):
        # Expand this layer's image-token scores to the processed image size.
        # Image-token vector [grid_size**2] -> [batch=1, channel=1, grid, grid].
        interpolation_input = scores.reshape(1, 1, grid_size, grid_size)
        if binary_mode:
            # Resize binary token selection [1, 1, grid, grid] to [height, width].
            heatmap = F.interpolate(
                interpolation_input,
                size=(image_height, image_width),
                mode="nearest",
            )[0, 0].numpy()
        else:
            # Resize continuous IGA [1, 1, grid, grid] to [height, width].
            heatmap = F.interpolate(
                interpolation_input,
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
        if not heatmap_only:
            axis.imshow(image)
        heatmap_view = axis.imshow(heatmap, cmap=color_map, norm=color_norm)
        if not heatmap_only:
            heatmap_view.set_alpha(0.5)
        layer_index = layer_name.split("_layers_")[-1].split("_", 1)[0]
        axis.set_title(f"Layer {layer_index}")

    for axis in flat_axes[panel_count:]:
        axis.axis("off")
    for axis in flat_axes[:panel_count]:
        axis.axis("off")
    if heatmap_view is not None:
        colorbar = figure.colorbar(
            heatmap_view,
            ax=flat_axes[:panel_count].tolist(),
            fraction=0.015,
            pad=0.01,
            label=(
                "Top IGA image-token mask (binary)"
                if binary_mode
                else (
                    "Attention over all keys (%) (log scale)"
                    if attention_percent
                    else "Attention weight (log scale)"
                )
            ),
        )
        if binary_mode:
            colorbar.set_ticks([0, 1])
            colorbar.set_ticklabels(["Other tokens", "Retained top tokens"])
    if binary_mode:
        display_title = f"top {top_iga_percent:.1%} IGA image tokens per layer"
    elif attention_percent:
        display_title = "all-key attention percentage"
    else:
        display_title = "image-key attention weight"
    figure.suptitle(
        f"Calibration sample {sample_number}: {display_title}"
    )
    plt.show()
    plt.close(figure)


# ============================================================================
# CLI AND CONFIGURATION
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse dataset, attention display, and logging options."""
    parser = argparse.ArgumentParser(
        description="Run the configured COCO calibration subset and capture LLaVA attention weights."
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
    parser.add_argument("--seed", type=int, default=None, help="Override configured seed.")
    parser.add_argument(
        "--log-level",
        choices=("none", "normal", "extensive"),
        default="normal",
        help="Output detail: none, concise progress, or detailed progress and timings.",
    )
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress Python and dependency warnings.")
    parser.add_argument(
        "--mode",
        choices=("online", "save"),
        default="online",
        help="online: display one IGA heatmap per sample without saving maps; save: save raw maps for later.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print elapsed time for setup steps, each forward pass, and each heatmap display.",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Show standalone heatmaps instead of overlays on the image.",
    )
    attention_display = parser.add_mutually_exclusive_group()
    attention_display.add_argument(
        "--true-attention-percent",
        action="store_true",
        help=(
            "Display each image token's softmax attention as a percentage of attention over all valid keys "
            "instead of as a decimal weight."
        ),
    )
    attention_display.add_argument(
        "--top-iga-percent",
        "--top-token-percent",
        dest="top_iga_percent",
        type=float,
        default=None,
        help="Show a binary map retaining this fraction of highest-IGA image tokens per layer (0.1 = 10%).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the five phases: dataset, model, hooks, sample forwards, and heatmaps."""
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    try:
        dataset_config = read_yaml(args.dataset_config)
        model_config = read_yaml(args.model_config)
        data_paths = get_data_paths(args.dataset_config)
        sample_count = args.samples if args.samples is not None else dataset_config.get("calibration_samples", 2)
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("calibration_samples must be a positive integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("calibration_seed must be an integer or null")
        if args.top_iga_percent is not None and not 0 < args.top_iga_percent <= 1:
            raise ValueError("--top-iga-percent must be in (0, 1], for example 0.1 for 10%")
        if args.top_iga_percent is not None and args.mode != "online":
            raise ValueError("--top-iga-percent requires --mode online to display the binary heatmaps")

        total_started = time.perf_counter() if args.timing else None

        # =====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # =====================================================================
        dataset_started = time.perf_counter() if args.timing else None
        dataset: Dataset = load_sharegpt4v_dataset(
            source="coco",
            config_path=args.dataset_config,
            download_images=True,
            existing_images_only=True,
        )
        if sample_count > len(dataset):
            raise ValueError(
                f"Requested {sample_count} samples, but only {len(dataset)} COCO records are available"
            )
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))
        log(
            f"Dataset source=COCO, selected={len(dataset):,}, requested={sample_count}, "
            f"seed={seed}, dataset_config={args.dataset_config}",
            level="extensive",
        )

        calibration_dir = repository_path(
            dataset_config.get(
                "calibration_output_dir",
                data_paths["processed_dir"] / "llava15_coco_calibration",
            )
        )
        calibration_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(calibration_dir))
        log(f"Saved {len(dataset):,} calibration records to {calibration_dir}")
        if dataset_started is not None:
            log(f"[timing] Prepare calibration dataset: {time.perf_counter() - dataset_started:.2f} s")

        # =====================================================================
        # 2. LOAD MODEL
        # =====================================================================
        model_started = time.perf_counter() if args.timing else None
        model_id = model_config["model_id"]
        model_dir = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        model_dir.mkdir(parents=True, exist_ok=True)
        log(f"Downloading or updating model {model_id} at {model_dir}")
        snapshot_download(repo_id=model_id, local_dir=str(model_dir))

        model, input_adapter = load_model(
            model_dir,
            model_config,
            repository_root=REPOSITORY_ROOT,
        )
        model.config.output_attentions = True
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
        # 3. REGISTER ATTENTION HOOKS
        # =====================================================================
        hooks_started = time.perf_counter() if args.timing else None
        attention_dir = repository_path(
            dataset_config.get(
                "attention_output_dir",
                data_paths["processed_dir"] / "attention_maps",
            )
        )
        if args.mode == "save":
            attention_dir.mkdir(parents=True, exist_ok=True)
        # Hook state schema: directory is the current save path or None;
        # captured counts matrices in the current forward; text_mask and
        # image_mask are boolean [sequence] masks; iga maps layer name to one
        # CPU [image_tokens] vector. Reset captured/iga before each sample.
        active_sample: dict[str, Any] = {
            "directory": None,
            "captured": 0,
            "text_mask": None,
            "image_mask": None,
            "iga": {},
        }
        hook_handles = register_attention_hooks(model, args.mode, active_sample)
        log(f"Registered hooks on {len(hook_handles)} self-attention layers")
        if hooks_started is not None:
            log(f"[timing] Register hooks: {time.perf_counter() - hooks_started:.2f} s")

        # =====================================================================
        # 4. FORWARD CALIBRATION SAMPLES
        # =====================================================================
        log(f"Forwarding {len(dataset):,} samples one at a time...")
        try:
            for index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                sample_dir = attention_dir / f"sample_{index:04d}" if args.mode == "save" else None
                if sample_dir is not None:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                active_sample["directory"] = sample_dir
                active_sample["captured"] = 0
                active_sample["iga"] = {}

                user_text = next(
                    (turn.get("value", "") for turn in sample["conversations"] if turn.get("from") in {"human", "user"}),
                    "",
                )
                assistant_text = next(
                    (turn.get("value", "") for turn in sample["conversations"] if turn.get("from") in {"gpt", "assistant"}),
                    "",
                )
                # Input row schema consumed by the QIG adapter: conversation
                # turns, stable sample id, and the local-image sentinel.
                row = {
                    "conversations": sample["conversations"],
                    "id": sample.get("id", str(index)),
                    "image": "local",
                }
                prepared = input_adapter.preprocess_data([sample["image"]], row)
                batch = input_adapter.data_collator([prepared])
                prompt_inputs, prompt_kwargs = input_adapter.generate_input(batch)
                # Adapter masks may arrive on different devices. Keep all
                # mask arithmetic on CPU; hooks move masks to attention-device.
                # Keep bookkeeping masks as CPU boolean [sequence] vectors;
                # hooks move them to the attention tensor device before indexing.
                attention_mask = prompt_kwargs["attention_mask"][0].bool().detach().cpu()
                image_mask = prompt_kwargs["vision_mask"][0].bool().detach().cpu()
                text_mask = attention_mask & ~image_mask
                if not image_mask.any() or not text_mask.any():
                    raise ValueError("Could not identify text-query and image-key token positions")
                # Image positions are [image_tokens]; use the final position to
                # restrict IGA query rows to text after the image.
                image_end = image_mask.nonzero(as_tuple=True)[0][-1]
                text_mask &= torch.arange(text_mask.numel()) > image_end
                if not text_mask.any():
                    raise ValueError("No text query tokens occur after the image tokens")
                log(
                    f"Sample {index} id={row['id']}: embeds={tuple(prompt_inputs['inputs_embeds'].shape)}, "
                    f"valid_tokens={int(attention_mask.sum())}, image_tokens={int(image_mask.sum())}, "
                    f"text_query_tokens={int(text_mask.sum())}",
                    level="extensive",
                )
                active_sample["image_mask"] = image_mask
                active_sample["text_mask"] = text_mask
                if sample_started is not None:
                    log(
                        f"[timing] Sample {index} preprocessing: "
                        f"{time.perf_counter() - sample_started:.2f} s",
                    )

                forward_started = time.perf_counter() if args.timing else None
                with torch.inference_mode():
                    outputs = input_adapter(
                        inputs_embeds=prompt_inputs["inputs_embeds"],
                        attention_mask=prompt_kwargs["attention_mask"],
                        labels=prompt_kwargs["labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                # Forward logits shape: [batch, sequence, vocabulary].
                if forward_started is not None:
                    log(f"[timing] Sample {index} forward: {time.perf_counter() - forward_started:.2f} s")
                if active_sample["captured"] == 0:
                    raise RuntimeError(
                        "Attention hooks captured no weights; check the Transformers attention implementation"
                    )
                log(f"[{index}/{len(dataset)}] logits {tuple(outputs.logits.shape)}", level="extensive")
                log(
                    f"[{index}/{len(dataset)}] captured_attention_layers={active_sample['captured']}",
                    level="extensive",
                )
                if args.mode == "online":
                    layer_scores = active_sample["iga"]
                    if not layer_scores:
                        raise RuntimeError("No per-layer IGA scores were captured")
                    # Combine per-layer [image_tokens] vectors into one
                    # [layers * image_tokens] vector for the sample summary.
                    score_values = torch.cat(list(layer_scores.values()))
                    log(
                        f"[{index}/{len(dataset)}] IGA layers={len(layer_scores)}, "
                        f"image-token scores min={score_values.min():.6g}, "
                        f"mean={score_values.mean():.6g}, max={score_values.max():.6g}",
                        level="extensive",
                    )
                    # Print the conversation before showing the image and every layer's map.
                    log(f"[{index}/{len(dataset)}] Conversation:", level="extensive")
                    log(f"  USER: {user_text}", level="extensive")
                    log(f"  ASSISTANT: {assistant_text}", level="extensive")
                    plot_started = time.perf_counter() if args.timing else None
                    show_iga_heatmap(
                        sample["image"],
                        layer_scores,
                        index,
                        heatmap_only=args.heatmap_only,
                        attention_percent=args.true_attention_percent,
                        top_iga_percent=args.top_iga_percent,
                    )
                    if plot_started is not None:
                        log(
                            f"[timing] Sample {index} heatmap display: "
                            f"{time.perf_counter() - plot_started:.2f} s",
                        )
                else:
                    log(
                        f"  Saved {active_sample['captured']} attention matrices to {sample_dir}",
                        level="extensive",
                    )
                del outputs, prompt_inputs, prompt_kwargs, batch, prepared, sample
        finally:
            for handle in hook_handles:
                handle.remove()

        # =====================================================================
        # 5. ANALYSE CAPTURED ATTENTION
        # =====================================================================
        # Online mode calculates IGA and displays the per-sample heatmap in the
        # forward loop. Add further saved-map analysis here.

        if args.mode == "online":
            log("Online IGA heatmap display complete; no attention maps were saved.")
        else:
            log(f"Attention maps saved under: {attention_dir}")
        if total_started is not None:
            log(f"[timing] Total: {time.perf_counter() - total_started:.2f} s")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
