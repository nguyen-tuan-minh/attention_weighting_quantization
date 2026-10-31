"""Evaluate an LVLM with the QIG companion LMMS-Eval installation."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_TASKS = "vizwiz_vqa_val,mmmu_val,chartqa,ai2d,scienceqa_img"
MEMORY_UNITS = {
    "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
}


def repository_path(value: str | Path) -> Path:
    """Resolve a user path from the repository root unless it is absolute."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def parse_memory_size(value: str) -> int:
    """Parse a model-placement limit; a unitless number means decimal GB."""
    import re

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*((?:[kmgt]i?b)|b)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError("use B, KB, MB, GB, TB, KiB, MiB, GiB, or TiB; bare numbers mean GB")
    unit = (match.group(2) or "GB").lower()
    return int(float(match.group(1)) * MEMORY_UNITS[unit])


def lmms_subprocess_environment() -> dict[str, str]:
    """Make the project and vendored evaluation packages importable by child Python processes."""
    qig_root = REPOSITORY_ROOT / "third_party" / "QIG"
    source_paths = (
        REPOSITORY_ROOT / "src",
        qig_root,
        qig_root / "3rdparty" / "LLaVA-NeXT",
        qig_root / "3rdparty" / "lmms-eval",
    )
    environment = os.environ.copy()
    existing_paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    python_paths = [str(path) for path in source_paths if path.is_dir()]
    python_paths.extend(path for path in existing_paths if path and path not in python_paths)
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


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
        "--contrastive-amateur", type=str, default=None,
        help="Amateur checkpoint; enables contrastive LMMS-Eval with the expert checkpoint in --pretrained.",
    )
    parser.add_argument(
        "--contrastive-weights", type=float, nargs="+", default=None,
        help="Evaluate each contrastive weight separately (requires --contrastive-amateur).",
    )
    parser.add_argument(
        "--contrastive-score-type", choices=("logits", "log-probabilities"), default="logits",
        help="Subtract raw logits or log probabilities during contrastive decoding.",
    )
    parser.add_argument(
        "--plausibility-threshold", type=float, default=0.01,
        help="Keep tokens whose expert probability is at least this fraction of the expert top-token probability.",
    )
    for role in ("expert", "amateur"):
        for tier in ("gpu", "cpu"):
            parser.add_argument(
                f"--max-{role}-model-memory-{tier}", type=parse_memory_size, default=None,
                help=f"Optional {role} model {tier.upper()} placement cap; bare numbers mean GB.",
            )
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


def run_lmms_eval_with_contrastive_decoding(arguments: list[str]) -> int:
    """Run LMMS-Eval with an in-memory LLaVA adapter patch; vendored files stay untouched."""
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
    import torch.nn.functional as functional
    from accelerate.utils import get_max_memory
    from lmms_eval.models import llava as llava_module
    from lmms_eval.models.llava import Llava

    original_init = Llava.__init__
    original_loader = llava_module.load_pretrained_model

    def placement_limits(gpu_limit: int | None, cpu_limit: int | None) -> dict | None:
        if gpu_limit is None and cpu_limit is None:
            return None
        limits = get_max_memory()
        if cpu_limit is not None:
            limits["cpu"] = cpu_limit
        if gpu_limit is not None:
            cuda_indices = [index for index in limits if isinstance(index, int)]
            if not cuda_indices:
                raise RuntimeError("A GPU model memory limit was set, but no CUDA device is available")
            limits[min(cuda_indices)] = gpu_limit
        return limits

    def init_with_contrastive(
        self,
        *model_args,
        contrastive_amateur: str,
        contrastive_weight: float,
        contrastive_score_type: str = "logits",
        plausibility_threshold: float = 0.01,
        contrastive_expert_gpu_memory: int | None = None,
        contrastive_expert_cpu_memory: int | None = None,
        contrastive_amateur_gpu_memory: int | None = None,
        contrastive_amateur_cpu_memory: int | None = None,
        contrastive_offload_dir: str | None = None,
        **model_kwargs,
    ) -> None:
        if int(model_kwargs.get("batch_size", 1)) != 1:
            raise ValueError("Contrastive LLaVA evaluation currently requires --batch-size 1")
        expert_path = str(model_kwargs.get("pretrained", model_args[0] if model_args else ""))
        if not expert_path:
            raise ValueError("Contrastive evaluation needs the expert checkpoint in pretrained=...")
        expert_limits = placement_limits(contrastive_expert_gpu_memory, contrastive_expert_cpu_memory)
        amateur_limits = placement_limits(contrastive_amateur_gpu_memory, contrastive_amateur_cpu_memory)
        offload_root = Path(contrastive_offload_dir or (REPOSITORY_ROOT / "outputs/evaluations/.contrastive-offload"))
        expert_offload = offload_root / "expert"
        amateur_offload = offload_root / "amateur"
        expert_offload.mkdir(parents=True, exist_ok=True)
        amateur_offload.mkdir(parents=True, exist_ok=True)

        def loader_with_placement(path, *loader_args, **loader_kwargs):
            path_text = str(path)
            role_limits = expert_limits if path_text == expert_path else amateur_limits
            role_folder = expert_offload if path_text == expert_path else amateur_offload
            loader_kwargs["device_map"] = "auto"
            loader_kwargs["offload_folder"] = str(role_folder)
            loader_kwargs["offload_state_dict"] = True
            if role_limits is not None:
                loader_kwargs["max_memory"] = role_limits
            return original_loader(path, *loader_args, **loader_kwargs)

        # Llava's constructor uses this imported builder. Wrap it only in this
        # process to set independent placement budgets for each checkpoint.
        llava_module.load_pretrained_model = loader_with_placement
        model_kwargs["device_map"] = "auto"
        original_init(self, *model_args, **model_kwargs)
        if not hasattr(self, "_world_size"):
            # The vendored adapter's auto-map branch uses _word_size by mistake.
            self._world_size = 1
        if self.world_size != 1:
            raise ValueError("Contrastive evaluation currently supports one LMMS-Eval process")

        from llava.mm_utils import get_model_name_from_path

        amateur_name = get_model_name_from_path(contrastive_amateur)
        amateur_tokenizer, amateur_model, amateur_processor, _ = loader_with_placement(
            contrastive_amateur,
            None,
            amateur_name,
            device_map="auto",
            multimodal=True,
            attn_implementation=model_kwargs.get("attn_implementation"),
        )
        if self.tokenizer.get_vocab() != amateur_tokenizer.get_vocab():
            raise ValueError("Expert and amateur tokenizers have different vocabularies")
        for name in ("size", "crop_size", "image_mean", "image_std"):
            if getattr(self._image_processor, name, None) != getattr(amateur_processor, name, None):
                raise ValueError(f"Expert and amateur image processors differ in {name}")

        amateur_model.eval()
        amateur_model.requires_grad_(False)
        self._contrastive_amateur_model = amateur_model
        self._contrastive_weight = float(contrastive_weight)
        self._contrastive_score_type = contrastive_score_type
        self._contrastive_plausibility_threshold = float(plausibility_threshold)
        if not math.isfinite(self._contrastive_weight) or self._contrastive_weight < 0:
            raise ValueError("contrastive_weight must be finite and non-negative")
        if not 0 < self._contrastive_plausibility_threshold <= 1:
            raise ValueError("plausibility_threshold must be in (0, 1]")

        expert_model = self.model
        amateur_model_ref = amateur_model
        weight = self._contrastive_weight
        score_type = self._contrastive_score_type
        threshold = self._contrastive_plausibility_threshold

        def contrastive_generate(
            _expert,
            input_ids,
            attention_mask=None,
            images=None,
            image_sizes=None,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            use_cache=True,
            **unused,
        ):
            if input_ids.shape[0] != 1 or num_beams != 1:
                raise ValueError("Contrastive decoding supports batch size 1 and num_beams=1")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            sequence = input_ids
            expert_past = amateur_past = None
            eos_value = getattr(expert_model.config, "eos_token_id", None)
            eos_ids = set(eos_value) if isinstance(eos_value, (list, tuple)) else ({eos_value} if eos_value is not None else set())
            temperature = float(temperature or 0.0)
            top_p = 1.0 if top_p is None else float(top_p)
            if do_sample and temperature > 0 and not 0 < top_p <= 1:
                raise ValueError("LMMS-Eval top_p must be in (0, 1] for probabilistic contrastive decoding")

            with torch.inference_mode():
                for _ in range(int(max_new_tokens)):
                    expert_cached = use_cache and expert_past is not None
                    amateur_cached = use_cache and amateur_past is not None
                    expert_input = sequence[:, -1:] if expert_cached else sequence
                    amateur_input = sequence[:, -1:] if amateur_cached else sequence
                    expert_output = expert_model(
                        input_ids=expert_input,
                        attention_mask=attention_mask,
                        past_key_values=expert_past if expert_cached else None,
                        images=images if not expert_cached else None,
                        image_sizes=image_sizes if not expert_cached else None,
                        use_cache=True,
                        return_dict=True,
                    )
                    amateur_output = amateur_model_ref(
                        input_ids=amateur_input,
                        attention_mask=attention_mask,
                        past_key_values=amateur_past if amateur_cached else None,
                        images=images if not amateur_cached else None,
                        image_sizes=image_sizes if not amateur_cached else None,
                        use_cache=True,
                        return_dict=True,
                    )
                    # LLaVA expands each image placeholder into visual tokens
                    # during the prefill. The caller's attention mask only
                    # covers the original text tokens, so rebuild it from the
                    # actual cache length before the next cached step.
                    if use_cache and expert_past is None:
                        expert_past = expert_output.past_key_values
                        amateur_past = amateur_output.past_key_values

                        def cache_length(cache):
                            if hasattr(cache, "get_seq_length"):
                                return int(cache.get_seq_length())
                            return int(cache[0][0].shape[-2])

                        expert_length = cache_length(expert_past)
                        amateur_length = cache_length(amateur_past)
                        if expert_length != amateur_length:
                            raise RuntimeError(
                                "Expert and amateur produced different KV sequence lengths "
                                f"({expert_length} vs {amateur_length}); check image preprocessing and prompt tokenization"
                            )
                        attention_mask = attention_mask.new_ones((input_ids.shape[0], expert_length))
                    expert_logits = expert_output.logits[:, -1, :].float()
                    amateur_logits = amateur_output.logits[:, -1, :].float().to(expert_logits.device)
                    if expert_logits.shape[-1] != amateur_logits.shape[-1]:
                        raise RuntimeError("Expert and amateur output vocabulary sizes differ")
                    expert_log_probs = functional.log_softmax(expert_logits[0], dim=-1)
                    expert_probabilities = expert_log_probs.exp()
                    allowed = expert_probabilities >= expert_probabilities.max() * threshold
                    if score_type == "logits":
                        expert_scores = expert_logits[0]
                        amateur_scores = amateur_logits[0]
                    else:
                        expert_scores = expert_log_probs
                        amateur_scores = functional.log_softmax(amateur_logits[0], dim=-1)
                    scores = (expert_scores - weight * amateur_scores).masked_fill(~allowed, -torch.inf)

                    if do_sample and temperature > 0:
                        probabilities = torch.softmax(scores / temperature, dim=-1)
                        sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
                        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                        remove = cumulative - sorted_probabilities >= top_p
                        probabilities[sorted_ids[remove]] = 0
                        probabilities /= probabilities.sum()
                        next_token = torch.multinomial(probabilities, num_samples=1).reshape(1, 1)
                    else:
                        next_token = scores.argmax(dim=-1).reshape(1, 1)
                    next_token = next_token.to(input_ids.device)
                    sequence = torch.cat((sequence, next_token), dim=1)
                    attention_mask = torch.cat(
                        (attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))), dim=1
                    )
                    expert_past = expert_output.past_key_values
                    amateur_past = amateur_output.past_key_values
                    if int(next_token.item()) in eos_ids:
                        break

            return sequence

        expert_model.generate = types.MethodType(contrastive_generate, expert_model)
        print(
            f"LMMS-Eval contrastive decoding: amateur={contrastive_amateur}; weight={weight}; "
            f"score_type={score_type}; plausibility_threshold={threshold}; "
            f"expert placement={getattr(expert_model, 'hf_device_map', 'single-device')}; "
            f"amateur placement={getattr(amateur_model, 'hf_device_map', 'single-device')}",
            flush=True,
        )

    Llava.__init__ = init_with_contrastive
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
    if len(sys.argv) > 1 and sys.argv[1] == "--_lmms-eval-contrastive-worker":
        return run_lmms_eval_with_contrastive_decoding(sys.argv[2:])

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

    if args.contrastive_weights is not None and args.contrastive_amateur is None:
        raise ValueError("--contrastive-weights requires --contrastive-amateur")
    if args.contrastive_amateur is None and any(
        getattr(args, f"max_{role}_model_memory_{tier}") is not None
        for role in ("expert", "amateur") for tier in ("gpu", "cpu")
    ):
        raise ValueError("The expert/amateur memory limits are only used with --contrastive-amateur")
    if args.contrastive_amateur is not None:
        if args.model != "llava":
            raise ValueError("Contrastive decoding is currently supported with --model llava only")
        if str(args.batch_size) != "1":
            raise ValueError("Contrastive LLaVA evaluation currently requires --batch-size 1")
        if args.remove_new_task_data:
            raise ValueError("--remove-new-task-data cannot be combined with contrastive weight sweeps")
        if not math.isfinite(args.plausibility_threshold) or not 0 < args.plausibility_threshold <= 1:
            raise ValueError("--plausibility-threshold must be in (0, 1]")
        contrastive_weights = args.contrastive_weights or [0.5]
        if any(not math.isfinite(weight) or weight < 0 for weight in contrastive_weights):
            raise ValueError("contrastive weights must be finite and non-negative")
        if len(set(contrastive_weights)) != len(contrastive_weights):
            raise ValueError("--contrastive-weights cannot contain duplicates")
        amateur_value = args.contrastive_amateur
        amateur_local_path = repository_path(amateur_value)
        amateur = str(amateur_local_path.resolve()) if amateur_local_path.exists() else amateur_value

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

    subprocess_environment = lmms_subprocess_environment()

    if args.contrastive_amateur is not None:
        if args.overlay_dir is not None:
            raise ValueError("Use a standalone expert checkpoint for contrastive evaluation; --overlay-dir is not supported here")
        for weight in contrastive_weights:
            weight_name = f"weight-{weight:g}"
            weight_output_dir = output_dir / weight_name
            weight_output_dir.mkdir(parents=True, exist_ok=True)
            run_model_args = [
                *model_args,
                f"contrastive_amateur={amateur}",
                f"contrastive_weight={weight}",
                f"contrastive_score_type={args.contrastive_score_type}",
                f"plausibility_threshold={args.plausibility_threshold}",
                f"contrastive_offload_dir={weight_output_dir / '.model-offload'}",
                "device_map=auto",
            ]
            for role in ("expert", "amateur"):
                for tier in ("gpu", "cpu"):
                    limit = getattr(args, f"max_{role}_model_memory_{tier}")
                    if limit is not None:
                        run_model_args.append(f"contrastive_{role}_{tier}_memory={limit}")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_lmms-eval-contrastive-worker",
                "--model", args.model,
                "--model_args", ",".join(run_model_args),
                "--batch_size", str(args.batch_size),
                "--output_path", str(weight_output_dir),
                "--seed", str(args.seed),
                "--verbosity", args.verbosity,
                "--tasks", args.tasks,
            ]
            if args.device:
                command.extend(["--device", args.device])
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            if args.log_samples:
                command.append("--log_samples")
            print(f"Expert model: {pretrained}", flush=True)
            print(f"Amateur model: {amateur}", flush=True)
            print(f"Contrastive weight: {weight}; score type: {args.contrastive_score_type}", flush=True)
            print(f"LMMS-Eval results: {weight_output_dir}", flush=True)
            result = subprocess.run(
                command, cwd=REPOSITORY_ROOT, env=subprocess_environment, check=False
            )
            if result.returncode != 0:
                return result.returncode
        print(f"Completed contrastive weight sweep: {', '.join(map(str, contrastive_weights))}", flush=True)
        return 0

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
                [*command, "--tasks", args.tasks], cwd=REPOSITORY_ROOT,
                env=subprocess_environment, check=False,
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
                    [*command, "--tasks", task], cwd=REPOSITORY_ROOT,
                    env=subprocess_environment, check=False,
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
