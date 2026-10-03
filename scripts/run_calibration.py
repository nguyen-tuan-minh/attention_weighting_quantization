"""Load the configured model and forward calibration samples individually."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import (  # noqa: E402
    get_data_paths,
    load_sharegpt4v_dataset,
)
from attention_quantization.models import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the configured model, prepare calibration data, and forward each sample."
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
        "--download-model-only",
        action="store_true",
        help="Prepare the dataset first, download the model, then exit without forwarding samples.",
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

        # ==================================================================
        # 1. PREPARE CALIBRATION DATASET
        # ==================================================================
        dataset_source = dataset_config.get("calibration_source", "coco")
        print(f"Loading ShareGPT4V {dataset_source} data and ensuring images are available...")
        dataset = load_sharegpt4v_dataset(
            source=dataset_source,
            config_path=args.dataset_config,
            download_images=True,
            existing_images_only=True,
        )
        if sample_count > len(dataset):
            raise ValueError(
                f"Requested {sample_count} samples, but only {len(dataset)} "
                f"{dataset_source} records are available"
            )
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))

        calibration_dir = repository_path(
            dataset_config.get(
                "calibration_output_dir",
                data_paths["processed_dir"] / "calibration_samples",
            )
        )
        calibration_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(calibration_dir))
        print(f"Saved {len(dataset):,} calibration records to {calibration_dir}")

        # ==================================================================
        # 2. DOWNLOAD AND LOAD MODEL
        # ==================================================================
        model_id = model_config["model_id"]
        model_dir = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading or updating model {model_id} at {model_dir}")
        snapshot_download(repo_id=model_id, local_dir=str(model_dir))
        if args.download_model_only:
            print("Dataset preparation and model download complete.")
            return 0

        model, input_adapter = load_model(
            model_dir,
            model_config,
            repository_root=REPOSITORY_ROOT,
        )

        # ==================================================================
        # 3. FORWARD CALIBRATION SAMPLES
        # ==================================================================
        print(f"Forwarding {len(dataset):,} samples one at a time...")
        for index, sample in enumerate(dataset, start=1):
            row = {"conversations": sample["conversations"], "id": sample.get("id", str(index)), "image": "local"}
            prepared = input_adapter.preprocess_data([sample["image"]], row)
            batch = input_adapter.data_collator([prepared])
            prompt_inputs, prompt_kwargs = input_adapter.generate_input(batch)

            with torch.inference_mode():
                outputs = input_adapter(
                    inputs_embeds=prompt_inputs["inputs_embeds"],
                    attention_mask=prompt_kwargs["attention_mask"],
                    labels=prompt_kwargs["labels"],
                    use_cache=False,
                    return_dict=True,
                )
            print(f"[{index}/{len(dataset)}] logits shape: {tuple(outputs.logits.shape)}")
            del outputs, prompt_inputs, prompt_kwargs, batch, prepared, sample

        print("Calibration forward passes complete.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
