"""Run QIG on ShareGPT4V COCO and save a reusable quantized checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.data import get_data_paths, load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.quantization.qig import load_qig_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize LLaVA 1.5 with the QIG source implementation."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the checkpoint and QIG metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory after a successful run.")
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--model-config", type=Path, default=REPOSITORY_ROOT / "configs" / "model.yaml")
    parser.add_argument("--base-model", default=None, help="Local checkpoint or Hugging Face model ID; defaults to configs/model.yaml.")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--w-group", type=int, default=128)
    parser.add_argument(
        "--method",
        choices=("qig", "mbq", "awq", "smoothquant", "rtn", "gptq"),
        default="qig",
        help="Quantization method supplied by QIG's qmllm package.",
    )
    parser.add_argument("--a-bit", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant scaling parameter.")
    parser.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damping parameter.")
    parser.add_argument("--loss-mode", choices=("mae", "mse"), default="mae")
    parser.add_argument("--reweight", action="store_true", help="Enable QIG's additional gradient-based modality reweighting.")
    parser.add_argument("--distort", action="store_true", help="Enable QIG's feature distortion option where supported.")
    return parser.parse_args()


def qig_source_path() -> Path:
    value = os.environ.get("QIG_SOURCE_DIR", str(REPOSITORY_ROOT / ".third_party" / "QIG"))
    return Path(value).expanduser().resolve()


def qig_git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_base_model(model_reference: str) -> str:
    """Resolve local relative checkpoints from the project root, not QIG's cwd."""
    model_path = Path(model_reference).expanduser()
    if model_path.is_absolute():
        return str(model_path.resolve())

    project_model_path = (REPOSITORY_ROOT / model_path).resolve()
    if project_model_path.exists():
        return str(project_model_path)
    # Keep Hub IDs such as ``liuhaotian/llava-v1.5-7b`` unchanged.
    return model_reference


def has_local_model_weights(model_dir: Path) -> bool:
    has_weights = any(model_dir.glob("pytorch_model*.bin")) or any(
        model_dir.glob("model*.safetensors")
    )
    return (model_dir / "config.json").is_file() and has_weights


def write_qig_calibration_jsonl(
    dataset_config_path: Path,
    sample_count: int,
    seed: int | None,
    destination: Path,
) -> tuple[int, list[str]]:
    """Export the sampled ShareGPT4V COCO rows in QIG's expected JSONL format."""
    from datasets import Image

    paths = get_data_paths(dataset_config_path)
    dataset = load_sharegpt4v_dataset(
        source="coco",
        config_path=dataset_config_path,
        # QIG should use the COCO files already downloaded by the data
        # workflow; do not start another large ZIP download during quantization.
        download_images=False,
    ).cast_column("image", Image(decode=False))
    if sample_count < 1:
        raise ValueError("--samples must be a positive integer")

    image_root = paths["raw_dir"].resolve()

    def image_is_available(row: dict[str, Any]) -> bool:
        image_value = row.get("image")
        image_path = image_value.get("path") if isinstance(image_value, dict) else None
        if not isinstance(image_path, str) or not image_path:
            return False
        resolved_image = Path(image_path).expanduser().resolve()
        try:
            resolved_image.relative_to(image_root)
        except ValueError:
            return False
        return resolved_image.is_file()

    total_records = len(dataset)
    dataset = dataset.filter(
        image_is_available,
        desc="Keeping ShareGPT4V COCO records with local images",
    )
    available_records = len(dataset)
    missing_records = total_records - available_records
    print(
        f"COCO records with images on disk: {available_records:,}/{total_records:,} "
        f"({missing_records:,} records skipped)"
    )
    if sample_count > available_records:
        raise ValueError(
            f"Requested {sample_count} samples, but only {available_records} COCO records "
            "have images available locally"
        )

    if seed is None:
        dataset = dataset.shuffle()
    else:
        dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(sample_count))

    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_ids: list[str] = []
    with destination.open("w", encoding="utf-8") as output:
        for sample_index, row in enumerate(dataset):
            image_value: Any = row.get("image")
            image_path = image_value.get("path") if isinstance(image_value, dict) else None
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("Could not read the local image path from a COCO sample")
            resolved_image = Path(image_path).expanduser().resolve()
            try:
                relative_image = resolved_image.relative_to(image_root)
            except ValueError as error:
                raise ValueError(f"Image path {resolved_image} is outside configured raw_dir {image_root}") from error
            if not resolved_image.is_file():
                raise FileNotFoundError(
                    f"COCO image disappeared after local-image filtering: {resolved_image}"
                )

            conversations = row.get("conversations")
            if not isinstance(conversations, list):
                raise ValueError("ShareGPT4V sample has no 'conversations' list")
            record = {
                "id": str(row.get("id", sample_index)),
                "image": relative_image.as_posix(),
                "conversations": conversations,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            selected_ids.append(record["id"])
    return len(dataset), selected_ids


def publish_directory(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    backup_dir: Path | None = None
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}; pass --overwrite to replace it")
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
        backup_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
        backup_dir.rmdir()
        os.replace(output_dir, backup_dir)

    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def main() -> int:
    args = parse_args()
    if (args.reweight or args.distort) and args.method not in {"qig", "mbq"}:
        raise ValueError("--reweight and --distort are supported only by the qig and mbq methods")
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir in {Path("/"), REPOSITORY_ROOT.resolve()}:
        raise ValueError(f"Refusing unsafe output directory: {output_dir}")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}; pass --overwrite to replace it")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    qig_source = qig_source_path()
    if not (qig_source / "main_quant.py").is_file():
        raise FileNotFoundError(f"QIG source was not found at {qig_source}; run scripts/quantize_qig.sh first")
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch CUDA installation first.")

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        calibration_path: Path | None = None
        if args.method == "rtn":
            sample_count, selected_sample_ids = 0, []
        else:
            calibration_path = staging_dir / "calibration.jsonl"
            sample_count, selected_sample_ids = write_qig_calibration_jsonl(
                args.dataset_config,
                args.samples,
                args.seed,
                calibration_path,
            )

        # Load and run QIG through this project's isolated integration module.
        import torch

        qig = load_qig_runtime(qig_source)
        model_class = qig.get_model("llava")
        # QIG's classic LLaVA adapter does not accept a `dtype` argument.
        # Its upstream LLaVA builder defaults to loading the model in float16.
        # QIG's LLaVA builder infers the model family from the checkpoint's
        # final path component. The project's local directory name does not
        # match its LLaVA 1.5 detection pattern, so
        # provide the recognized architecture name explicitly while loading
        # weights from the requested local path.
        model_config = read_yaml(args.model_config)
        if args.base_model is None:
            model_id = model_config.get("model_id", "liuhaotian/llava-v1.5-7b")
            model_dir = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
            if not has_local_model_weights(model_dir):
                from huggingface_hub import snapshot_download

                model_dir.mkdir(parents=True, exist_ok=True)
                print(f"Downloading original LLaVA checkpoint {model_id} to {model_dir}")
                snapshot_download(repo_id=model_id, local_dir=str(model_dir))
            base_model = str(model_dir.resolve())
        else:
            base_model = resolve_base_model(args.base_model)
        attn_implementation = model_config.get("attn_implementation", "eager")
        model_args = (
            f"pretrained={base_model},model_name=llava-v1.5-7b,"
            f"attn_implementation={attn_implementation}"
        )
        lm = model_class.create_from_arg_string(
            model_args,
            {"batch_size": 1, "device": "cuda"},
        )
        process_class = qig.get_process_model("llava")
        process_model = process_class(
            lm._model,
            lm._tokenizer,
            getattr(lm, "processor", None),
        )

        if calibration_path is None:
            prompt_inputs, prompt_kwargs = None, None
        else:
            data_paths = get_data_paths(args.dataset_config)
            prompt_inputs, prompt_kwargs = qig.get_multimodal_calib_dataset(
                data_path=str(calibration_path),
                image_folder=str(data_paths["raw_dir"]),
                model=process_model,
                n_samples=sample_count,
                micro_bs=args.micro_batch_size,
            )

        scales_path = (
            None
            if args.method in {"rtn", "gptq"}
            else staging_dir / f"{args.method}_scales.pt"
        )
        qig_args = argparse.Namespace(
            method=args.method,
            run_process=True,
            pseudo_quant=True,
            scale_path=str(scales_path) if scales_path is not None else None,
            w_group=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            alpha=args.alpha,
            reweight=args.reweight,
            distort=args.distort,
            loss_mode=args.loss_mode,
            percdamp=args.percdamp,
            model="llava",
            model_args=model_args,
        )
        qig.qwrapper(process_model, prompt_inputs, prompt_kwargs, qig_args)

        # QIG uses pseudo quantization: the weights are rounded onto the
        # requested grid but stored here in their normal floating-point dtype.
        lm._model.save_pretrained(
            str(staging_dir),
            safe_serialization=True,
            max_shard_size="5GB",
        )
        lm._tokenizer.save_pretrained(str(staging_dir))
        image_processor = getattr(process_model, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "save_pretrained"):
            image_processor.save_pretrained(str(staging_dir))

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": args.method,
            "model_type": "llava",
            "base_model": base_model,
            "model_dtype": model_config.get("torch_dtype", "float16"),
            "weight_bits": args.w_bit,
            "activation_bits": args.a_bit,
            "group_size": args.w_group,
            "reweight": args.reweight,
            "distort": args.distort,
            "loss_mode": args.loss_mode,
            "alpha": args.alpha,
            "gptq_percdamp": args.percdamp,
            "calibration_samples": sample_count,
            "calibration_seed": args.seed,
            "calibration_data": "ShareGPT4V COCO" if calibration_path is not None else None,
            "calibration_sample_ids": selected_sample_ids,
            "calibration_jsonl": calibration_path.name if calibration_path is not None else None,
            "scale_file": scales_path.name if scales_path is not None else None,
            "weight_storage": "floating point pseudo-quantized weights; not packed low-bit weights",
            "qig_source": str(qig_source),
            "qig_git_revision": qig_git_revision(qig_source),
            "llava_source_revision": qig_git_revision(qig_source / "3rdparty" / "LLaVA-NeXT"),
            "lmms_eval_source_revision": qig_git_revision(qig_source / "3rdparty" / "lmms-eval"),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        (staging_dir / "qig_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "README.txt").write_text(
            "QIG LLaVA 1.5 quantization artifact.\n"
            "The model weights are pseudo-quantized values stored in floating point format.\n"
            "This checkpoint is for research/reproduction and is not a packed INT4 deployment model.\n"
            "Load it with attention_quantization.models.load_qig_quantized_model().\n",
            encoding="utf-8",
        )

        publish_directory(staging_dir, output_dir, args.overwrite)
        print(f"Saved QIG quantization artifact to: {output_dir}")
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError) as error:
        if os.environ.get("ATTENTION_QUANTIZATION_DEBUG") == "1":
            traceback.print_exc()
        else:
            print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
