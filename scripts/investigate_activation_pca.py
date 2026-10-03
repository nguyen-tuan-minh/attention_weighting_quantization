"""Visualize image and text token activations with layer-wise PCA."""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.models import load_model  # noqa: E402


def register_layernorm_hooks(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture each decoder layer's input after its input LayerNorm."""
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.input_layernorm$", name)
        if match is None or "vision_tower" in name:
            continue
        layer_index = int(match.group(1))

        def capture_activation(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            index: int = layer_index,
        ) -> None:
            activation = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(activation) or activation.ndim != 3:
                return
            # Keep only one sample on CPU; do not retain the forward graph or
            # accumulate all layer activations on the GPU.
            active_sample["activations"][index] = activation[0].detach().to(
                device="cpu", dtype=torch.float16
            )

        handles.append(module.register_forward_hook(capture_activation))

    if not handles:
        raise RuntimeError("No decoder input_layernorm modules were found in the loaded model")
    return handles


def layerwise_pca(
    activations: dict[int, torch.Tensor],
    image_mask: torch.Tensor,
    text_mask: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Project the combined image and text token cloud to two PCA components."""
    projected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_index, activation in sorted(activations.items()):
        if activation.shape[0] != image_mask.numel() or activation.shape[0] != text_mask.numel():
            raise ValueError(
                f"Layer {layer_index} has {activation.shape[0]} token positions, "
                f"but masks have {image_mask.numel()} and {text_mask.numel()}"
            )

        selected_mask = image_mask | text_mask
        values = activation[selected_mask].float()
        labels = image_mask[selected_mask]
        if values.shape[0] < 3:
            raise ValueError(f"Layer {layer_index} has too few image/text tokens for 2D PCA")

        # Randomized low-rank PCA avoids forming a hidden_size x hidden_size
        # covariance matrix for every transformer layer.
        scores, singular_values, _ = torch.pca_lowrank(
            values,
            q=2,
            center=True,
            niter=2,
        )
        projected[layer_index] = (scores * singular_values, labels)
    return projected


def show_activation_pca(
    projected: dict[int, tuple[torch.Tensor, torch.Tensor]],
    sample_number: int,
    save_dir: Path | None = None,
) -> None:
    """Display one image/text PCA scatter plot for every decoder layer."""
    if not projected:
        raise ValueError("No normalized layer activations were captured")

    layer_indices = sorted(projected)
    columns = 6
    rows = math.ceil(len(layer_indices) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3.3, rows * 2.8),
        constrained_layout=True,
        squeeze=False,
    )

    image_color = "#2878B5"
    text_color = "#E87500"
    for axis, layer_index in zip(axes.ravel(), layer_indices):
        coordinates, is_image = projected[layer_index]
        image_points = coordinates[is_image].numpy()
        text_points = coordinates[~is_image].numpy()
        axis.scatter(
            image_points[:, 0],
            image_points[:, 1],
            s=8,
            alpha=0.55,
            color=image_color,
            label="Image tokens",
            rasterized=True,
        )
        axis.scatter(
            text_points[:, 0],
            text_points[:, 1],
            s=15,
            alpha=0.8,
            color=text_color,
            label="Text tokens",
            rasterized=True,
        )
        axis.set_title(f"Layer {layer_index}")
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
        axis.tick_params(labelsize=7)

    for axis in axes.ravel()[len(layer_indices):]:
        axis.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.suptitle(
        f"Sample {sample_number}: post-input-LayerNorm activations",
        y=1.02,
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"sample_{sample_number:04d}_activation_pca.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved PCA figure to {output_path}")
    plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LLaVA image and text token activations with layer-wise PCA."
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
        "--save-dir",
        type=Path,
        default=None,
        help="Optionally save each sample's PCA figure to this directory.",
    )
    parser.add_argument("--timing", action="store_true", help="Print elapsed time for each stage and sample.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
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

        total_started = time.perf_counter() if args.timing else None

        # =====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # =====================================================================
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
        dataset = dataset.shuffle(seed=seed).select(range(sample_count))
        print(f"Selected {len(dataset):,} COCO samples with local images")
        if dataset_started is not None:
            print(f"[timing] Prepare calibration dataset: {time.perf_counter() - dataset_started:.2f} s")

        # =====================================================================
        # 2. LOAD MODEL
        # =====================================================================
        model_started = time.perf_counter() if args.timing else None
        model_id = model_config["model_id"]
        model_dir = repository_path(model_config.get("model_dir", "models/llava-1.5-7b"))
        has_weights = any(model_dir.glob("pytorch_model*.bin")) or any(
            model_dir.glob("model*.safetensors")
        )
        if not (model_dir / "config.json").is_file() or not has_weights:
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading LLaVA checkpoint {model_id} to {model_dir}")
            snapshot_download(repo_id=model_id, local_dir=str(model_dir))

        model, input_adapter = load_model(
            model_dir,
            model_config,
            repository_root=REPOSITORY_ROOT,
        )
        model.eval()
        if model_started is not None:
            print(f"[timing] Load model: {time.perf_counter() - model_started:.2f} s")

        # =====================================================================
        # 3. REGISTER HOOKS
        # =====================================================================
        hooks_started = time.perf_counter() if args.timing else None
        active_sample: dict[str, Any] = {"activations": {}}
        hook_handles = register_layernorm_hooks(model, active_sample)
        print(f"Registered hooks on {len(hook_handles)} decoder input LayerNorm modules")
        if hooks_started is not None:
            print(f"[timing] Register hooks: {time.perf_counter() - hooks_started:.2f} s")

        # =====================================================================
        # 4. FORWARD CALIBRATION SAMPLES
        # =====================================================================
        print(f"Forwarding {len(dataset):,} samples one at a time...")
        try:
            for index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                active_sample["activations"] = {}
                row = {
                    "conversations": sample["conversations"],
                    "id": sample.get("id", str(index)),
                    "image": "local",
                }
                prepared = input_adapter.preprocess_data([sample["image"]], row)
                batch = input_adapter.data_collator([prepared])
                prompt_inputs, prompt_kwargs = input_adapter.generate_input(batch)

                attention_mask = prompt_kwargs["attention_mask"][0].bool().detach().cpu()
                image_mask = prompt_kwargs["vision_mask"][0].bool().detach().cpu()
                text_mask = attention_mask & ~image_mask
                if not image_mask.any() or not text_mask.any():
                    raise ValueError("Could not identify image and text token positions")

                user_text = next(
                    (
                        turn.get("value", "")
                        for turn in sample["conversations"]
                        if turn.get("from") in {"human", "user"}
                    ),
                    "",
                )
                assistant_text = next(
                    (
                        turn.get("value", "")
                        for turn in sample["conversations"]
                        if turn.get("from") in {"gpt", "assistant"}
                    ),
                    "",
                )
                print(f"[{index}/{len(dataset)}] Conversation:")
                print(f"  USER: {user_text}")
                print(f"  ASSISTANT: {assistant_text}")

                forward_started = time.perf_counter() if args.timing else None
                with torch.inference_mode():
                    outputs = input_adapter(
                        inputs_embeds=prompt_inputs["inputs_embeds"],
                        attention_mask=prompt_kwargs["attention_mask"],
                        labels=prompt_kwargs["labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                if forward_started is not None:
                    print(f"[timing] Sample {index} forward: {time.perf_counter() - forward_started:.2f} s")
                if not active_sample["activations"]:
                    raise RuntimeError("No layernorm hooks captured activations")

                # =============================================================
                # 5. ANALYSE ACTIVATIONS FOR THIS SAMPLE
                # =============================================================
                # Run PCA before the next forward pass, then release this
                # sample's activations instead of retaining all samples in RAM.
                analyse_started = time.perf_counter() if args.timing else None
                projected = layerwise_pca(
                    active_sample["activations"],
                    image_mask,
                    text_mask,
                )
                print(f"[{index}/{len(dataset)}] PCA calculated for {len(projected)} layers")
                show_activation_pca(projected, index, args.save_dir)
                if analyse_started is not None:
                    print(f"[timing] Sample {index} PCA and plot: {time.perf_counter() - analyse_started:.2f} s")
                if sample_started is not None:
                    print(f"[timing] Sample {index} total: {time.perf_counter() - sample_started:.2f} s")
                del outputs, prompt_inputs, prompt_kwargs, batch, prepared, projected, sample
        finally:
            for handle in hook_handles:
                handle.remove()

        print("Layer-wise image/text activation PCA complete.")
        if total_started is not None:
            print(f"[timing] Total: {time.perf_counter() - total_started:.2f} s")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
