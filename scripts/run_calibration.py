"""Download LLaVA, prepare a COCO subset, and forward samples individually."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LLaVA 1.5 7B, prepare calibration data, and forward each sample."
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
    parser.add_argument("--download-model-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset_config = read_yaml(args.dataset_config)
        model_config = read_yaml(args.model_config)
        data_paths = get_data_paths(args.dataset_config)

        sample_count = args.samples if args.samples is not None else dataset_config.get("calibration_samples", 128)
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("calibration_samples must be a positive integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("calibration_seed must be an integer or null")

        model_id = model_config.get("model_id", "llava-hf/llava-1.5-7b-hf")
        model_dir = repository_path(model_config.get("model_dir", "models/llava-1.5-7b"))
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading or updating model {model_id} at {model_dir}")
        snapshot_download(repo_id=model_id, local_dir=str(model_dir))
        if args.download_model_only:
            print("Model download complete.")
            return 0

        print("Loading COCO ShareGPT4V data and ensuring COCO images are available...")
        dataset = load_sharegpt4v_dataset(
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

        processor = AutoProcessor.from_pretrained(str(model_dir))
        dtype_name = model_config.get("torch_dtype", "float16")
        model_dtype = getattr(torch, dtype_name, None)
        if model_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("torch_dtype must be float16, bfloat16, or float32")
        device_map = model_config.get("device_map", "auto")
        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=model_dtype,
            device_map=device_map,
        )
        model.eval()

        print(f"Forwarding {len(dataset):,} samples one at a time...")
        for index, sample in enumerate(dataset, start=1):
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

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
            print(f"[{index}/{len(dataset)}] logits shape: {tuple(outputs.logits.shape)}")
            del outputs, inputs, encoded, sample

        print("Calibration forward passes complete.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
