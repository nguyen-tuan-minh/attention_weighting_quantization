"""Evaluate an LVLM with the QIG companion LMMS-Eval installation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_TASKS = "vizwiz_vqa_val,mmmu_val,chartqa,ai2d,scienceqa_img"


def repository_path(value: str | Path) -> Path:
    """Resolve a user path from the repository root unless it is absolute."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark evaluation through the QIG companion LMMS-Eval package. "
            "The default task set follows the five benchmarks used in the QIG paper."
        )
    )
    parser.add_argument("--model-config", type=Path, default=REPOSITORY_ROOT / "configs" / "model.yaml")
    parser.add_argument("--pretrained", type=str, default=None, help="Model ID or local checkpoint directory; defaults to configs/model.yaml.")
    parser.add_argument(
        "--overlay-dir", type=Path, default=None,
        help="Apply a packed GPTQ layer overlay to the base checkpoint during evaluation.",
    )
    parser.add_argument("--model", default="llava", help="LMMS-Eval model adapter name (default: llava).")
    parser.add_argument("--model-args", default="", help="Additional comma-separated LMMS-Eval model arguments.")
    parser.add_argument("--tasks", default=PAPER_TASKS, help=f"Comma-separated LMMS-Eval task IDs (default: {PAPER_TASKS}).")
    parser.add_argument("--batch-size", default="1", help="LMMS-Eval batch size; use 1 for LLaVA 1.5 compatibility.")
    parser.add_argument("--device", default=None, help="Evaluation device, for example cuda:0.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Result directory (default: outputs/evaluations/<timestamp>).")
    parser.add_argument("--limit", type=float, default=None, help="Limit examples per task for a smoke run; do not use for reported metrics.")
    parser.add_argument("--seed", default="42", help="LMMS-Eval seed (single integer or four comma-separated seeds).")
    parser.add_argument("--log-samples", action="store_true", help="Save per-example prompts, predictions, and targets.")
    parser.add_argument(
        "--remove-new-task-data",
        action="store_true",
        help=(
            "Run each task separately, then remove Hugging Face cache files created by that task. "
            "Files present before the task and the model's Hub cache are preserved."
        ),
    )
    parser.add_argument("--verbosity", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_lmms_eval_with_overlay(arguments: list[str]) -> int:
    """Patch LMMS-Eval's LLaVA constructor in-process to apply a packed overlay."""
    qig_root = REPOSITORY_ROOT / "third_party" / "QIG"
    for source in (
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "src",
        qig_root,
        qig_root / "3rdparty" / "LLaVA-NeXT",
        qig_root / "3rdparty" / "lmms-eval",
    ):
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))

    import torch
    from lmms_eval.models.llava import Llava
    from scripts.quantize.quantize_gptq import (
        dequantize_entry,
        linear_layers,
        register_after_weights_loaded,
    )

    original_init = Llava.__init__

    def init_with_overlay(self, *model_args, overlay_dir: str, **model_kwargs):
        original_init(self, *model_args, **model_kwargs)
        layer_dir = Path(overlay_dir) / "layers"
        overlay_files = sorted(layer_dir.glob("decoder-*.pt"))
        if not overlay_files:
            raise FileNotFoundError(f"No decoder overlay files found under {layer_dir}")

        handles = []
        model = self.model
        applied = 0
        for index, block in enumerate(model.model.layers):
            overlay_path = layer_dir / f"decoder-{index:03d}.pt"
            if not overlay_path.is_file():
                continue
            state = torch.load(overlay_path, map_location="cpu", weights_only=True)
            linears = linear_layers(block)

            def restore(_module, _inputs, state=state, linears=linears):
                for name, entry in state.items():
                    parameter = linears[name].weight
                    parameter.data.copy_(dequantize_entry(entry, parameter.device, parameter.dtype))

            accelerate_hook = getattr(block, "_hf_hook", None)
            if accelerate_hook is None or not hasattr(accelerate_hook, "pre_forward"):
                # Resident weights persist across generation, so dequantize once.
                restore(block, ())
            else:
                handles.append(register_after_weights_loaded(block, restore))
            applied += 1

        if applied != len(overlay_files):
            raise ValueError(
                f"Found {len(overlay_files)} layer overlays but applied {applied}; "
                "the overlay may belong to a different model architecture"
            )
        self._gptq_overlay_handles = handles
        print(f"Applied {applied} GPTQ decoder layer overlays from {overlay_dir}", flush=True)

    Llava.__init__ = init_with_overlay
    sys.argv = [sys.argv[0], *arguments]
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()
    return 0


def cache_directories() -> tuple[Path, Path]:
    """Return the Hugging Face Hub and datasets cache locations for this process."""
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", str(hf_home / "datasets"))).expanduser()
    return hub_cache, datasets_cache


def cache_inventory(root: Path, preserve_subtree: Path | None = None) -> tuple[set[Path], set[Path]]:
    """List files and directories under a cache, optionally excluding a subtree."""
    if not root.exists():
        return set(), set()
    files: set[Path] = set()
    directories: set[Path] = {root}
    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        if preserve_subtree is not None:
            try:
                current.relative_to(preserve_subtree)
                child_directories[:] = []
                continue
            except ValueError:
                child_directories[:] = [
                    child for child in child_directories if (current / child) != preserve_subtree
                ]
        directories.add(current)
        for filename in filenames:
            files.add(current / filename)
        # Directory symlinks are listed by os.walk but aren't traversed. Preserve them.
        for child in child_directories[:]:
            child_path = current / child
            if child_path.is_symlink():
                files.add(child_path)
                child_directories.remove(child)
            else:
                directories.add(child_path)
    return files, directories


def remove_new_cache_files(
    roots: Iterable[tuple[Path, set[Path], set[Path]]],
    hub_root: Path,
    preserved_hub_subtree: Path | None,
) -> list[Path]:
    """Remove files created since the snapshots and then prune empty cache directories."""
    removed: list[Path] = []
    for root, initial_files, initial_directories in roots:
        current_files, _ = cache_inventory(root, preserved_hub_subtree if root == hub_root else None)
        for path in current_files - initial_files:
            try:
                path.unlink()
                removed.append(path)
            except FileNotFoundError:
                pass
        if root.exists():
            for directory, child_directories, filenames in os.walk(root, topdown=False, followlinks=False):
                current = Path(directory)
                if preserved_hub_subtree is not None and root == hub_root:
                    try:
                        current.relative_to(preserved_hub_subtree)
                        continue
                    except ValueError:
                        pass
                # Don't remove directories that existed before this task, even if empty.
                if not filenames and not child_directories and current not in initial_directories:
                    try:
                        current.rmdir()
                    except OSError:
                        pass
    return removed


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--_lmms-eval-overlay-worker":
        return run_lmms_eval_with_overlay(sys.argv[2:])

    args = parse_args()
    model_config_path = repository_path(args.model_config)
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_config_path}")
    with model_config_path.open(encoding="utf-8") as config_file:
        model_config = yaml.safe_load(config_file) or {}

    pretrained = args.pretrained or model_config.get("model_dir") or model_config.get("model_id")
    if not pretrained:
        raise ValueError(f"No model path configured in {model_config_path}; pass --pretrained.")
    pretrained_path = Path(pretrained).expanduser()
    if pretrained_path.exists():
        pretrained = str(pretrained_path.resolve())

    overlay_dir = None
    if args.overlay_dir is not None:
        if args.model != "llava":
            raise ValueError("--overlay-dir is supported with the llava model adapter")
        overlay_dir = repository_path(args.overlay_dir)
        metadata_path = overlay_dir / "gptq_metadata.json"
        layers_dir = overlay_dir / "layers"
        if not metadata_path.is_file() or not layers_dir.is_dir():
            raise FileNotFoundError(f"Expected GPTQ overlay metadata and layers under {overlay_dir}")
        with metadata_path.open(encoding="utf-8") as metadata_file:
            overlay_metadata = yaml.safe_load(metadata_file) or {}
        if overlay_metadata.get("format") != "attention-quantization-gptq-packed-overlay-v1":
            raise ValueError(f"Unsupported GPTQ overlay format in {metadata_path}")

    output_dir = args.output_dir or Path("outputs/evaluations") / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = repository_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = [f"pretrained={pretrained}"]
    if overlay_dir is not None:
        model_args.append(f"overlay_dir={overlay_dir}")
    for key in ("device_map", "attn_implementation"):
        value = model_config.get(key)
        if value is not None:
            model_args.append(f"{key}={value}")
    if args.model_args:
        model_args.append(args.model_args.strip().strip(","))

    command = [sys.executable]
    if overlay_dir is None:
        command.extend(["-m", "lmms_eval"])
    else:
        command.extend([str(Path(__file__).resolve()), "--_lmms-eval-overlay-worker"])
    command.extend([
        "--model",
        args.model,
        "--model_args",
        ",".join(model_args),
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--verbosity",
        args.verbosity,
    ])
    if args.device:
        command.extend(["--device", args.device])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.log_samples:
        command.append("--log_samples")

    print(f"Model: {pretrained}", flush=True)
    if overlay_dir is not None:
        print(f"GPTQ overlay: {overlay_dir}", flush=True)
    print(f"LMMS-Eval results: {output_dir}", flush=True)
    print("LMMS-Eval reports each task's configured metric; metrics differ by benchmark.", flush=True)
    try:
        if not args.remove_new_task_data:
            print(f"Tasks: {args.tasks}", flush=True)
            return subprocess.run(
                [*command, "--tasks", args.tasks], cwd=REPOSITORY_ROOT, check=False
            ).returncode

        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        if not tasks:
            raise ValueError("--tasks must contain at least one task when cleanup is enabled.")
        hub_cache, datasets_cache = cache_directories()
        model_cache_subtree = None
        if "/" in pretrained and not Path(pretrained).exists():
            model_cache_name = "models--" + pretrained.strip("/").replace("/", "--")
            model_cache_subtree = hub_cache / model_cache_name

        for task_index, task in enumerate(tasks, start=1):
            print(f"Task {task_index}/{len(tasks)}: {task}", flush=True)
            hub_files, hub_directories = cache_inventory(hub_cache, model_cache_subtree)
            dataset_files, dataset_directories = cache_inventory(datasets_cache)
            cache_state = [
                (hub_cache, hub_files, hub_directories),
                (datasets_cache, dataset_files, dataset_directories),
            ]
            try:
                task_result = subprocess.run(
                    [*command, "--tasks", task], cwd=REPOSITORY_ROOT, check=False
                )
            finally:
                removed = remove_new_cache_files(cache_state, hub_cache, model_cache_subtree)
            print(f"Removed {len(removed)} newly created cache files for {task}.", flush=True)
            if task_result.returncode != 0:
                return task_result.returncode
        return 0
    except KeyboardInterrupt:
        print("Evaluation interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
