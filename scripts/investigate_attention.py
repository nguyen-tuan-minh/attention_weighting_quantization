"""Run calibration examples and analyse or save language attention weights."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import yaml
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, LlavaForConditionalGeneration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.data import get_data_paths, load_sharegpt4v_dataset  # noqa: E402


def read_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def conversation_text(sample: dict[str, Any]) -> tuple[str, str]:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("ShareGPT4V sample has no 'conversations' list")

    user_text = None
    assistant_text = None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        text = message.get("value", message.get("content"))
        if not isinstance(text, str):
            continue
        if role in {"human", "user"} and user_text is None:
            user_text = text.replace("<image>", "").strip()
        elif role in {"gpt", "assistant"} and user_text is not None:
            assistant_text = text.strip()
            break

    if not user_text or assistant_text is None:
        raise ValueError("ShareGPT4V sample needs a user prompt and assistant caption")
    return user_text, assistant_text


def register_attention_hooks(
    model: LlavaForConditionalGeneration,
    mode: str,
    active_sample: dict[str, Any],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Compute online IGA vectors or write each matrix for later analysis."""
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
                torch.save(
                    attention_weights.detach().to(device="cpu", dtype=torch.float16),
                    output_dir / f"{key}.pt",
                )
            else:
                text_mask = active_sample["text_mask"].to(attention_weights.device)
                image_mask = active_sample["image_mask"].to(attention_weights.device)
                # Shape after selection: heads x text queries x image keys.
                text_to_image = attention_weights[0][:, text_mask, :][:, :, image_mask]
                active_sample["iga"][key] = text_to_image.float().mean(dim=(0, 1)).cpu()
            active_sample["captured"] += 1

        handles.append(module.register_forward_hook(save_attention))

    if not handles:
        raise RuntimeError("No text self-attention modules were found in the loaded model")
    return handles


def show_iga_heatmap(
    encoded: Any,
    processor: Any,
    iga_scores: torch.Tensor,
    sample_number: int,
    layer_count: int,
) -> None:
    """Display the processed image beside its layer-averaged IGA overlay."""
    token_count = iga_scores.numel()
    grid_size = math.isqrt(token_count)
    if grid_size * grid_size != token_count:
        raise ValueError(
            f"Cannot reshape {token_count} image-token scores into a square patch grid"
        )

    pixel_values = encoded["pixel_values"][0].detach().cpu().float()
    image_processor = processor.image_processor
    mean = torch.tensor(image_processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(image_processor.image_std).view(-1, 1, 1)
    image = (pixel_values * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    image_height, image_width = image.shape[:2]
    heatmap = F.interpolate(
        iga_scores.reshape(1, 1, grid_size, grid_size),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Processor image")
    axes[1].imshow(image)
    heatmap_view = axes[1].imshow(heatmap, cmap="inferno", alpha=0.5)
    axes[1].set_title(f"IGA overlay (mean of {layer_count} layers)")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(heatmap_view, ax=axes[1], fraction=0.046, pad=0.04, label="IGA")
    figure.suptitle(f"Calibration sample {sample_number}")
    plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
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
        "--mode",
        choices=("online", "save"),
        default="online",
        help="online: display one IGA heatmap per sample without saving maps; save: save raw maps for later.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

        # =====================================================================
        # 1. LOAD MODEL
        # =====================================================================
        model_id = model_config.get("model_id", "llava-hf/llava-1.5-7b-hf")
        model_dir = repository_path(model_config.get("model_dir", "models/llava-1.5-7b"))
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading or updating model {model_id} at {model_dir}")
        snapshot_download(repo_id=model_id, local_dir=str(model_dir))

        dtype_name = model_config.get("torch_dtype", "float16")
        model_dtype = getattr(torch, dtype_name, None)
        if model_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("torch_dtype must be float16, bfloat16, or float32")
        processor = AutoProcessor.from_pretrained(str(model_dir))
        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=model_dtype,
            device_map=model_config.get("device_map", "auto"),
            attn_implementation=model_config.get("attn_implementation", "eager"),
        )
        model.eval()

        # =====================================================================
        # 2. PREPARE CALIBRATION DATASET
        # =====================================================================
        dataset: Dataset = load_sharegpt4v_dataset(
            source="coco",
            config_path=args.dataset_config,
            download_images=True,
        )
        if sample_count > len(dataset):
            raise ValueError(
                f"Requested {sample_count} samples, but only {len(dataset)} COCO records are available"
            )
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))

        calibration_dir = repository_path(
            dataset_config.get(
                "calibration_output_dir",
                data_paths["processed_dir"] / "llava15_coco_calibration",
            )
        )
        calibration_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(calibration_dir))
        print(f"Saved {len(dataset):,} calibration records to {calibration_dir}")

        # =====================================================================
        # 3. REGISTER ATTENTION HOOKS
        # =====================================================================
        attention_dir = repository_path(
            dataset_config.get(
                "attention_output_dir",
                data_paths["processed_dir"] / "attention_maps",
            )
        )
        if args.mode == "save":
            attention_dir.mkdir(parents=True, exist_ok=True)
        active_sample: dict[str, Any] = {
            "directory": None,
            "captured": 0,
            "text_mask": None,
            "image_mask": None,
            "iga": {},
        }
        hook_handles = register_attention_hooks(model, args.mode, active_sample)
        print(f"Registered hooks on {len(hook_handles)} self-attention layers")

        # =====================================================================
        # 4. FORWARD CALIBRATION SAMPLES
        # =====================================================================
        print(f"Forwarding {len(dataset):,} samples one at a time...")
        try:
            for index, sample in enumerate(dataset, start=1):
                sample_dir = attention_dir / f"sample_{index:04d}" if args.mode == "save" else None
                if sample_dir is not None:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                active_sample["directory"] = sample_dir
                active_sample["captured"] = 0
                active_sample["iga"] = {}

                user_text, assistant_text = conversation_text(sample)
                prompt = f"USER: <image>\n{user_text}\nASSISTANT: {assistant_text}"
                encoded = processor(
                    text=prompt,
                    images=sample["image"],
                    return_tensors="pt",
                )
                inputs = {key: value.to(model.device) for key, value in encoded.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                image_token_id = getattr(model.config, "image_token_id", None)
                if image_token_id is None:
                    image_token_id = model.config.image_token_index
                image_mask = inputs["input_ids"][0] == image_token_id
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    text_mask = ~image_mask
                else:
                    text_mask = attention_mask[0].bool() & ~image_mask
                if not image_mask.any() or not text_mask.any():
                    raise ValueError("Could not identify text-query and image-key token positions")
                # Ignore any prefix tokens before the image: causal attention
                # prevents those queries from attending to image keys.
                last_image_position = image_mask.nonzero(as_tuple=True)[0][-1]
                text_mask &= torch.arange(text_mask.numel(), device=text_mask.device) > last_image_position
                if not text_mask.any():
                    raise ValueError("No text query tokens occur after the image tokens")
                active_sample["image_mask"] = image_mask
                active_sample["text_mask"] = text_mask

                with torch.inference_mode():
                    outputs = model(**inputs, use_cache=False, output_attentions=True)
                if active_sample["captured"] == 0:
                    raise RuntimeError(
                        "Attention hooks captured no weights; check the Transformers attention implementation"
                    )
                print(f"[{index}/{len(dataset)}] logits {tuple(outputs.logits.shape)}")
                if args.mode == "online":
                    layer_scores = list(active_sample["iga"].values())
                    if not layer_scores:
                        raise RuntimeError("No per-layer IGA scores were captured")
                    iga_scores = torch.stack(layer_scores).mean(dim=0)
                    show_iga_heatmap(
                        encoded,
                        processor,
                        iga_scores,
                        index,
                        len(layer_scores),
                    )
                else:
                    print(
                        f"  Saved {active_sample['captured']} attention matrices to {sample_dir}"
                    )
                del outputs, inputs, encoded, sample
        finally:
            for handle in hook_handles:
                handle.remove()

        # =====================================================================
        # 5. ANALYSE CAPTURED ATTENTION
        # =====================================================================
        # Online mode calculates IGA and displays the per-sample heatmap in the
        # forward loop. Add further saved-map analysis here.

        if args.mode == "online":
            print("Online IGA heatmap display complete; no attention maps were saved.")
        else:
            print(f"Attention maps saved under: {attention_dir}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
