"""Disk-backed, layerwise GPTQ that writes a standalone LLaVA 1.5 checkpoint.

The requested output directory contains a complete dense FP16 model. GPTQ
weights are stored in a temporary packed form during quantization, then applied
to the selected decoder Linear layers before the complete checkpoint is saved.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import re
import shutil
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import get_data_paths, load_sharegpt4v_dataset  # noqa: E402

LOG_LEVEL = "normal"
SHOW_TIMING = False
RUN_STARTED: float | None = None
TIMING_SAMPLES: dict[str, list[float]] = {}


def record_timing(stage: str, elapsed: float) -> None:
    """Keep elapsed samples for the extensive end-of-run timing summary."""
    TIMING_SAMPLES.setdefault(stage, []).append(elapsed)


def log_timing_summary() -> None:
    """Report accumulated stage timing, including partial work on process exit."""
    if LOG_LEVEL != "extensive":
        return
    log("Timing summary (stage totals may overlap when one includes another):")
    for stage, samples in TIMING_SAMPLES.items():
        total = sum(samples)
        average = total / len(samples)
        log(
            f"Timing {stage}: count={len(samples)}, total={total:.1f}s, "
            f"avg={average:.2f}s, min={min(samples):.2f}s, max={max(samples):.2f}s"
        )
    if RUN_STARTED is not None:
        log(f"Timing total process wall time: {time.monotonic() - RUN_STARTED:.1f}s")


def log(message: str, *, level: str = "normal") -> None:
    levels = {"none": 0, "normal": 1, "extensive": 2}
    if levels[level] > levels[LOG_LEVEL]:
        return
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] [gptq] {message}", flush=True)


def summarize_device_map(device_map: dict[str, Any]) -> str:
    """Summarize how many modules Accelerate stores on each device."""
    counts: dict[str, int] = {}
    for location in device_map.values():
        key = str(location)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{location}: {count} modules" for location, count in counts.items())


def checkpoint_weight_summary(checkpoint: Path) -> tuple[int, list[tuple[str, int]]]:
    """Return local checkpoint weight-file sizes, largest first."""
    weight_files = [
        path for path in checkpoint.iterdir()
        if path.is_file() and (
            path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
            or path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
        )
    ]
    sizes = sorted(((path.name, path.stat().st_size) for path in weight_files), key=lambda item: item[1], reverse=True)
    return sum(size for _, size in sizes), sizes


def cuda_memory_summary() -> str:
    """Return allocated and reserved CUDA memory in GiB for progress logs."""
    if not torch.cuda.is_available():
        return "CUDA memory unavailable"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"CUDA memory allocated/reserved={allocated:.2f}/{reserved:.2f} GiB"


_MEMORY_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_memory_size(value: str) -> int:
    """Parse a memory limit to bytes; an omitted unit means decimal GB."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*((?:[kmgt]i?b)|b)?\s*", value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid memory size {value!r}; use a number with B, KB, MB, GB, TB, "
            "KiB, MiB, GiB, or TiB (a bare number means GB)"
        )
    try:
        amount = float(match.group(1))
        unit = (match.group(2) or "GB").lower()
        return int(amount * _MEMORY_UNITS[unit])
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(f"invalid memory size: {value!r}") from error


def format_memory_gb(memory_bytes: int | None) -> str:
    """Format bytes as decimal GB for human-readable diagnostics."""
    return "unlimited" if memory_bytes is None else f"{memory_bytes / 10**9:.2f} GB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the standalone quantized model checkpoint.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml", help="Model YAML configuration.")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset.yaml", help="Dataset YAML configuration.")
    parser.add_argument("--base-model", default=None, help="Override model.yaml's local checkpoint path.")
    parser.add_argument("--samples", type=int, default=None, help="Override dataset.yaml calibration_samples.")
    parser.add_argument("--seed", type=int, default=None, help="Override dataset.yaml calibration_seed.")
    parser.add_argument(
        "--log-level", choices=("none", "normal", "extensive"), default="normal",
        help="none suppresses progress; extensive adds resolved settings and execution diagnostics.",
    )
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress noisy Python and dependency warnings.")
    parser.add_argument("--timing", action="store_true", help="Show timing details at normal verbosity.")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), default=4, help="Weight precision in bits.")
    parser.add_argument("--group-size", type=int, default=128, help="Input columns per quantization group.")
    parser.add_argument(
        "--layers", default=None,
        help="1-based decoder layers to quantize, as comma-separated numbers/ranges (for example 1-4,8); default: all.",
    )
    parser.add_argument(
        "--hessian-group-layers", type=int, default=7,
        help="Selected decoder layers kept on GPU together for a complete pass over calibration samples.",
    )
    parser.add_argument(
        "--hessian-token-scope", choices=("all", "text"), default="all",
        help="Accumulate Hessians from all valid tokens or only non-image text tokens.",
    )
    parser.add_argument(
        "--hessian-weighting", choices=("uniform", "iga"), default="uniform",
        help=(
            "Use uniform token weights (default), or weight image-token activation rows by per-layer IGA; "
            "IGA requires --hessian-token-scope all."
        ),
    )
    parser.add_argument("--damp", type=float, default=0.01, help="Relative diagonal damping for Hessian factorization.")
    parser.add_argument(
        "--max-hessian-memory-gpu", type=parse_memory_size, default=None,
        help="Maximum CUDA memory for one active Hessian group; accepts KB/MB/GB/TB or KiB/MiB/GiB/TiB (bare number means GB).",
    )
    parser.add_argument(
        "--max-hessian-memory-cpu", type=parse_memory_size, default=None,
        help="Maximum system RAM for stored Hessians; excess uses disk. Units KB/MB/GB/TB or KiB/MiB/GiB/TiB; bare number means GB.",
    )
    parser.add_argument(
        "--max-hessian-memory-disk", type=parse_memory_size, default=None,
        help="Maximum disk space for stored Hessians. Units KB/MB/GB/TB or KiB/MiB/GiB/TiB; bare number means GB.",
    )
    parser.add_argument(
        "--max-model-memory-gpu", type=parse_memory_size, default=None,
        help="Optional GPU weight-placement limit; supports KB/MB/GB/TB or KiB/MiB/GiB/TiB (bare number means GB).",
    )
    parser.add_argument(
        "--max-model-memory-cpu", type=parse_memory_size, default=None,
        help="Optional CPU weight-placement limit; supports KB/MB/GB/TB or KiB/MiB/GiB/TiB (bare number means GB).",
    )
    parser.add_argument("--offload-dir", type=Path, default=None, help="Scratch disk directory; defaults under output-dir.")
    parser.add_argument("--keep-hessians", action="store_true", help="Keep disk-backed Hessian files after success for inspection.")
    parser.add_argument("--overwrite", action="store_true", help="Remove and recreate an existing output directory.")
    return parser.parse_args()


def load_runtime(qig_root: Path):
    """Import LLaVA's model builder and QIG's input adapter, not its quantizer."""
    llava_root = qig_root / "3rdparty/LLaVA-NeXT"
    for path in (qig_root, llava_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from llava.model.builder import load_pretrained_model
    from qmllm.models import get_process_model

    return load_pretrained_model, get_process_model


def resolve_checkpoint(args: argparse.Namespace, model_config: dict[str, Any]) -> Path:
    """Resolve and validate the local base-model checkpoint path."""
    configured = args.base_model or model_config.get("model_dir")
    if not configured:
        raise ValueError("Set model_dir in model.yaml or pass --base-model")
    checkpoint = repository_path(configured)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {checkpoint / 'config.json'}")
    return checkpoint


def sample_records(args: argparse.Namespace, data_config: dict[str, Any], dataset_config_path: Path):
    """Load local-image calibration records and return the configured random subset."""
    source = data_config.get("calibration_source", "coco")
    data = load_sharegpt4v_dataset(
        source=source,
        config_path=dataset_config_path,
        download_images=True,
        existing_images_only=True,
    )
    if not len(data):
        raw_dir = get_data_paths(dataset_config_path)["raw_dir"]
        raise ValueError(
            f"No ShareGPT4V {source!r} records match local images under {raw_dir}. "
            "Check the extracted image paths and raw_dir in the dataset config."
        )
    if args.samples < 1 or args.samples > len(data):
        raise ValueError(f"--samples must be in [1, {len(data)}]")
    return data.shuffle(seed=args.seed).select(range(args.samples))


def linear_layers(block: nn.Module) -> dict[str, nn.Linear]:
    """Return decoder Linear modules keyed by their names within one block."""
    return {name: module for name, module in block.named_modules() if isinstance(module, nn.Linear)}


class _MethodHookHandle:
    """Restore a wrapped Accelerate hook method when its temporary callback ends."""

    def __init__(self, owner: Any, method_name: str, original: Any, wrapper: Any):
        self.owner = owner
        self.method_name = method_name
        self.original = original
        self.wrapper = wrapper

    def remove(self) -> None:
        if getattr(self.owner, self.method_name, None) is self.wrapper:
            setattr(self.owner, self.method_name, self.original)


def register_after_weights_loaded(
    module: nn.Module,
    callback: Any,
):
    """Run a callback after Accelerate materializes offloaded weights, else as a pre-hook.

    PyTorch module pre-hooks execute before Accelerate's wrapped ``forward``.
    Wrapping the Accelerate hook's ``pre_forward`` places the callback after
    CPU/disk weights are loaded and before the module's actual forward call.
    """
    accelerate_hook = getattr(module, "_hf_hook", None)
    if accelerate_hook is None or not hasattr(accelerate_hook, "pre_forward"):
        return module.register_forward_pre_hook(callback)

    original = accelerate_hook.pre_forward

    def pre_forward_with_callback(loaded_module, *args, **kwargs):
        prepared = original(loaded_module, *args, **kwargs)
        callback(loaded_module, args)
        return prepared

    accelerate_hook.pre_forward = pre_forward_with_callback
    return _MethodHookHandle(accelerate_hook, "pre_forward", original, pre_forward_with_callback)


def parse_layer_selection(value: str | None, layer_count: int) -> set[int]:
    """Parse 1-based decoder layer numbers/ranges into zero-based indexes."""
    if value is None:
        return set(range(layer_count))
    selected: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            raise ValueError(f"Invalid --layers item {part!r}; use numbers and ranges such as 1-4,8")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > layer_count:
            raise ValueError(f"--layers range {part!r} must be within 1-{layer_count}")
        selected.update(range(start - 1, end))
    if not selected:
        raise ValueError("--layers must select at least one decoder layer")
    return selected


def gpu_hessian_hook(
    active_hessians: dict[str, torch.Tensor],
    width: int,
    counts: dict[str, int],
    key: str,
    token_selection: dict[str, torch.Tensor | None],
):
    """Capture a Linear input and accumulate its selected-token X'X on CUDA.

    ``token_selection['mask']`` is CPU bool [batch, sequence] or None. It is
    moved to the activation device before indexing. ``counts[key]`` accumulates
    selected activation rows across samples for this decoder Linear.
    """
    def capture(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0].detach().reshape(-1, width).float()
        mask = token_selection["mask"]
        if mask is not None:
            flat_mask = mask.reshape(-1)
            if flat_mask.numel() != x.shape[0]:
                raise RuntimeError(
                    f"Hessian token mask has {flat_mask.numel()} positions but {key} received {x.shape[0]} tokens"
                )
            x = x[flat_mask.to(device=x.device)]
            if x.shape[0] == 0:
                return
        counts[key] = counts.get(key, 0) + x.shape[0]
        hessian = active_hessians.get(key)
        if hessian is None:
            raise RuntimeError(f"GPU Hessian for {key} is not loaded for the active decoder group")
        x = x.to(device=hessian.device, dtype=torch.float32)
        hessian.addmm_(x.T, x)

    return capture


def accumulate_weighted_hessian(
    active_hessians: dict[str, torch.Tensor],
    counts: dict[str, int],
    key: str,
    width: int,
    activations: torch.Tensor,
    valid_mask: torch.Tensor,
    token_weights: torch.Tensor,
) -> None:
    """Add X'WX to a GPU Hessian for valid sequence rows."""
    x = activations.detach().reshape(-1, width).float()
    valid = valid_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
    weights = token_weights.reshape(-1).to(device=x.device, dtype=torch.float32)
    if valid.numel() != x.shape[0] or weights.numel() != x.shape[0]:
        raise RuntimeError(
            f"IGA masks/weights for {key} do not match its activation rows: "
            f"rows={x.shape[0]}, valid={valid.numel()}, weights={weights.numel()}"
        )
    x = x[valid]
    weights = weights[valid]
    if x.shape[0] == 0:
        return
    hessian = active_hessians.get(key)
    if hessian is None:
        raise RuntimeError(f"GPU Hessian for {key} is not loaded for the active decoder group")
    counts[key] = counts.get(key, 0) + x.shape[0]
    weighted_x = x * weights.unsqueeze(1)
    hessian.addmm_(x.T, weighted_x)


def make_iga_token_weights(
    attention_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_mask: torch.Tensor,
) -> torch.Tensor:
    """Return unit text weights and mean-one image-key weights from repository IGA."""
    if attention_weights.ndim != 4 or attention_weights.shape[0] != 1:
        raise RuntimeError(
            f"IGA expects post-softmax attention shaped [1, heads, query, key], "
            f"got {tuple(attention_weights.shape)}"
        )
    device = attention_weights.device
    valid = attention_mask.reshape(-1).to(device=device, dtype=torch.bool)
    image = vision_mask.reshape(-1).to(device=device, dtype=torch.bool)
    if valid.numel() != attention_weights.shape[-1] or image.numel() != valid.numel():
        raise RuntimeError("IGA attention length does not match the adapter's token masks")
    image_positions = image & valid
    image_end = torch.where(image_positions)[0][-1]
    positions = torch.arange(valid.numel(), device=device)
    text_queries = valid & ~image & (positions > image_end)

    # Match investigate_attention.py: mean post-softmax attention over heads
    # and post-image text queries, yielding one score for each image key.
    scores = attention_weights[0][:, text_queries, :][:, :, image_positions]
    iga = scores.float().mean(dim=(0, 1))
    # Softmax attention is finite and nonnegative; clamp the denominator to
    # avoid a host synchronization merely to check an all-zero edge case.
    mean_iga = iga.mean().clamp_min(torch.finfo(torch.float32).tiny)

    token_weights = torch.ones(valid.numel(), device=device, dtype=torch.float32)
    token_weights[image_positions] = iga / mean_iga
    return token_weights


def allocate_hessians(
    linears_by_layer: list[dict[str, nn.Linear]],
    hessian_dir: Path,
    ram_budget_bytes: int | None,
    disk_budget_bytes: int | None,
    selected_layers: set[int],
) -> tuple[dict[str, np.ndarray], dict[str, Path], dict[str, str], dict[str, tuple[int, int]], int, int]:
    """Allocate Hessians in RAM, then in one contiguous disk-backed float32 file."""
    entries = [
        (f"{layer_index}:{name}", linear)
        for layer_index, linears in enumerate(linears_by_layer) if layer_index in selected_layers
        for name, linear in linears.items()
    ]
    planned: list[tuple[str, nn.Linear, int, bool]] = []
    remaining_ram = ram_budget_bytes
    disk_bytes = 0
    for key, linear in entries:
        matrix_bytes = linear.in_features * linear.in_features * np.dtype(np.float32).itemsize
        in_ram = remaining_ram is None or matrix_bytes <= remaining_ram
        if in_ram:
            if remaining_ram is not None:
                remaining_ram -= matrix_bytes
        else:
            disk_bytes += matrix_bytes
        planned.append((key, linear, matrix_bytes, in_ram))

    hessian_dir.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(hessian_dir).free
    if free_disk < disk_bytes * 1.1:
        raise OSError(
            f"Disk-backed Hessians need about {disk_bytes / 1024**3:.1f} GiB plus temporary space; "
            f"only {free_disk / 1024**3:.1f} GiB is free at {hessian_dir}"
        )
    if disk_budget_bytes is not None and disk_bytes > disk_budget_bytes:
        raise OSError(
            f"Disk-backed Hessians need {format_memory_gb(disk_bytes)}, exceeding "
            f"--max-hessian-memory-disk={format_memory_gb(disk_budget_bytes)}"
        )

    hessians: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    placement: dict[str, str] = {}
    offsets: dict[str, tuple[int, int]] = {}
    disk_path = hessian_dir / "hessians.f32"
    disk_path.unlink(missing_ok=True)
    if disk_bytes:
        with disk_path.open("wb") as backing_file:
            backing_file.truncate(disk_bytes)
    disk_offset = 0
    for key, linear, _matrix_bytes, in_ram in planned:
        if in_ram:
            hessians[key] = np.zeros((linear.in_features, linear.in_features), dtype=np.float32)
            placement[key] = "ram"
            continue
        hessians[key] = np.memmap(
            disk_path, mode="r+", dtype=np.float32, offset=disk_offset,
            shape=(linear.in_features, linear.in_features),
        )
        paths[key] = disk_path
        offsets[key] = (disk_offset, linear.in_features**2)
        disk_offset += _matrix_bytes
        placement[key] = "disk"
    ram_bytes = (
        sum(matrix_bytes for _, _, matrix_bytes, in_ram in planned if in_ram)
        if ram_budget_bytes is None
        else ram_budget_bytes - remaining_ram
    )
    log(
        f"Hessian storage: {ram_bytes / 1024**3:.2f} GiB RAM, "
        f"{disk_bytes / 1024**3:.2f} GiB disk"
    )
    if disk_offset != disk_bytes:
        raise RuntimeError(f"Hessian disk layout mismatch: allocated {disk_offset} of {disk_bytes} bytes")
    if disk_bytes:
        log(f"Disk Hessians use one contiguous file: {disk_path} ({disk_bytes / 1024**3:.2f} GiB)")
    return hessians, paths, placement, offsets, ram_bytes, disk_bytes


def release_hessian(key: str, hessians: dict[str, np.ndarray], paths: dict[str, Path], keep: bool) -> None:
    """Release one matrix after it has been quantized; optionally delete its file."""
    hessian = hessians.pop(key, None)
    if isinstance(hessian, np.memmap):
        # Group-save code has already flushed the updated backing file. GPTQ
        # reads this map but never modifies it, so flushing each matrix again
        # here would add redundant disk synchronization.
        hessian._mmap.close()
    path = paths.get(key)
    if path is not None and not keep:
        paths.pop(key, None)
        if path not in paths.values():
            path.unlink(missing_ok=True)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack row-major quantized integers into a little-endian bit stream."""
    values = codes.detach().to(device="cpu", dtype=torch.int64).reshape(-1).numpy()
    packed = np.zeros((values.size * bits + 7) // 8, dtype=np.uint8)
    bit_positions = np.arange(values.size, dtype=np.int64) * bits
    byte_positions, offsets = bit_positions // 8, bit_positions % 8
    np.bitwise_or.at(packed, byte_positions, ((values << offsets) & 0xFF).astype(np.uint8))
    crosses = offsets + bits > 8
    if np.any(crosses):
        np.bitwise_or.at(
            packed,
            byte_positions[crosses] + 1,
            (values[crosses] >> (8 - offsets[crosses])).astype(np.uint8),
        )
    return torch.from_numpy(packed.copy())


def unpack_codes(packed: bytes, shape: tuple[int, int], bits: int, device: torch.device) -> torch.Tensor:
    count = shape[0] * shape[1]
    raw = packed.detach().cpu().numpy() if torch.is_tensor(packed) else np.frombuffer(packed, dtype=np.uint8)
    bit_positions = np.arange(count, dtype=np.int64) * bits
    byte_positions, offsets = bit_positions // 8, bit_positions % 8
    values = raw[byte_positions].astype(np.uint16) >> offsets
    crosses = offsets + bits > 8
    if np.any(crosses):
        values[crosses] |= raw[byte_positions[crosses] + 1].astype(np.uint16) << (8 - offsets[crosses])
    values &= (1 << bits) - 1
    return torch.from_numpy(values.astype(np.uint8).reshape(shape).copy()).to(device)


def dequantize_entry(entry: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows, columns = entry["shape"]
    bits, group_size = entry["bits"], entry["group_size"]
    codes = unpack_codes(entry["packed_weight"], (rows, columns), bits, device).float()
    scales = entry["scales"].to(device=device, dtype=torch.float32)
    zeros = entry["zeros"].to(device=device, dtype=torch.float32)
    result = torch.empty((rows, columns), device=device, dtype=torch.float32)
    for group_index, start in enumerate(range(0, columns, group_size)):
        end = min(start + group_size, columns)
        result[:, start:end] = (
            codes[:, start:end] - zeros[group_index, :, None]
        ) * scales[group_index, :, None]
    return result.to(dtype)


def quantize_linear_gptq(
    layer: nn.Linear,
    hessian: np.ndarray,
    token_count: int,
    *,
    bits: int,
    group_size: int,
    damp: float,
) -> dict[str, Any]:
    """Classic dense-Hessian GPTQ for one Linear; H is copied to GPU for use."""
    storage_device = layer.weight.device
    device = torch.device("cuda", torch.cuda.current_device())
    if storage_device != device:
        layer.weight.data = layer.weight.data.to(device)
    width = layer.in_features
    # Only this Linear's Hessian is resident on GPU during factorization.
    h = torch.from_numpy(np.array(hessian, copy=True)).to(device=device, dtype=torch.float32)
    h.mul_(2.0 / max(token_count, 1))
    dead = h.diagonal() == 0
    if dead.any():
        h[dead, dead] = 1
        layer.weight.data[:, dead] = 0
    h.diagonal().add_(damp * h.diagonal().mean())

    # GPTQ's inverse-Cholesky factor is computed in float64 for stability.
    h = h.double()
    chol = torch.linalg.cholesky(h)
    h_inv = torch.cholesky_inverse(chol)
    h_inv = torch.linalg.cholesky(h_inv, upper=True).float()
    del chol, h

    weight = layer.weight.data.float().clone()
    result = torch.zeros_like(weight)
    codes = torch.zeros((layer.out_features, width), device=device, dtype=torch.uint8)
    rows, columns = weight.shape
    qmax = (1 << bits) - 1
    group_size = columns if group_size <= 0 else group_size
    scales: list[torch.Tensor] = []
    zeros: list[torch.Tensor] = []
    for start in range(0, columns, group_size):
        group = weight[:, start : start + group_size]
        minimum, maximum = group.amin(dim=1, keepdim=True), group.amax(dim=1, keepdim=True)
        scale = (maximum - minimum).clamp_min(1e-8) / qmax
        zero = (-minimum / scale).round().clamp(0, qmax)
        scales.append(scale)
        zeros.append(zero)

    for block_start in range(0, columns, 128):
        block_end = min(block_start + 128, columns)
        w_block = weight[:, block_start:block_end].clone()
        q_block = torch.zeros_like(w_block)
        err_block = torch.zeros_like(w_block)
        h_block = h_inv[block_start:block_end, block_start:block_end]
        for offset in range(block_end - block_start):
            column = block_start + offset
            group_index = column // group_size
            w = w_block[:, offset]
            scale, zero = scales[group_index].squeeze(1), zeros[group_index].squeeze(1)
            q_code = ((w / scale) + zero).round().clamp(0, qmax)
            q = (q_code - zero) * scale
            q_block[:, offset] = q
            codes[:, column] = q_code.to(torch.uint8)
            diagonal = h_block[offset, offset].clamp_min(1e-8)
            error = (w - q) / diagonal
            err_block[:, offset] = error
            w_block[:, offset:] -= error.unsqueeze(1) * h_block[offset, offset:].unsqueeze(0)
        result[:, block_start:block_end] = q_block
        if block_end < columns:
            weight[:, block_end:] -= err_block @ h_inv[block_start:block_end, block_end:]
    layer.weight.data.copy_(result.to(dtype=layer.weight.dtype))
    entry = {
        "packed_weight": pack_codes(codes, bits),
        "scales": torch.stack([item.squeeze(1).cpu() for item in scales]),
        "zeros": torch.stack([item.squeeze(1).cpu() for item in zeros]),
        "shape": [layer.out_features, width],
        "bits": bits,
        "group_size": group_size,
    }
    if storage_device != device:
        layer.weight.data = layer.weight.data.to(storage_device)
    return entry


def main() -> int:
    """Collect grouped GPU Hessians, quantize selected layers, and save a standalone model."""
    global LOG_LEVEL, SHOW_TIMING, RUN_STARTED
    args = parse_args()
    LOG_LEVEL = args.log_level
    SHOW_TIMING = args.timing or args.log_level == "extensive"
    RUN_STARTED = time.monotonic()
    TIMING_SAMPLES.clear()
    if LOG_LEVEL == "extensive":
        atexit.register(log_timing_summary)
    if args.quiet_warnings:
        warnings.filterwarnings("ignore")
        for module_name in ("transformers.utils.logging", "datasets.utils.logging", "huggingface_hub.utils.logging"):
            try:
                module = __import__(module_name, fromlist=["set_verbosity_error"])
                module.set_verbosity_error()
            except ImportError:
                pass
    run_started = RUN_STARTED
    if args.group_size < 1:
        raise ValueError("--group-size must be positive")
    if args.hessian_group_layers < 1:
        raise ValueError("--hessian-group-layers must be positive")
    if args.hessian_weighting == "iga" and args.hessian_token_scope != "all":
        raise ValueError("--hessian-weighting iga requires --hessian-token-scope all")
    if args.damp < 0:
        raise ValueError("--damp must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA for layer execution and Hessian factorization")
    output_dir = repository_path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite")
    model_cfg = read_yaml(args.model_config)
    data_cfg = read_yaml(args.dataset_config)
    args.samples = args.samples if args.samples is not None else data_cfg.get("calibration_samples")
    args.seed = args.seed if args.seed is not None else data_cfg.get("calibration_seed")
    if args.samples is None:
        args.samples = 128
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    log(
        f"Configuration: bits={args.bits}, group_size={args.group_size}, samples={args.samples}, "
        f"seed={args.seed}, Hessian tokens={args.hessian_token_scope}, "
        f"Hessian weighting={args.hessian_weighting}, damp={args.damp}, "
        f"Hessian memory limits GPU/CPU/disk="
        f"{format_memory_gb(args.max_hessian_memory_gpu)}/"
        f"{format_memory_gb(args.max_hessian_memory_cpu)}/"
        f"{format_memory_gb(args.max_hessian_memory_disk)}"
    )
    log(
        f"Resolved configuration: model_config={args.model_config}, dataset_config={args.dataset_config}, "
        f"samples={args.samples}, seed={args.seed}",
        level="extensive",
    )
    checkpoint = resolve_checkpoint(args, model_cfg)
    dataset_started = time.monotonic()
    dataset = sample_records(args, data_cfg, args.dataset_config)
    record_timing("dataset loading and sample selection", time.monotonic() - dataset_started)
    log(f"Selected {len(dataset)} calibration records from source={data_cfg.get('calibration_source', 'coco')!r}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qig_root = repository_path(model_cfg.get("implementation_source_dir", "third_party/QIG"))
    runtime_started = time.monotonic()
    builder, get_process_model = load_runtime(qig_root)
    record_timing("runtime imports", time.monotonic() - runtime_started)
    offload_dir = repository_path(args.offload_dir) if args.offload_dir else output_dir / ".gptq-work"
    offload_dir.mkdir(parents=True, exist_ok=True)
    cleanup_default_scratch = args.offload_dir is None and not args.keep_hessians
    if cleanup_default_scratch:
        atexit.register(shutil.rmtree, offload_dir, ignore_errors=True)
    hessian_dir = offload_dir / "hessians"
    hessian_dir.mkdir(parents=True, exist_ok=True)
    # Packed decoder files are temporary intermediates; output_dir is the final
    # standalone LLaVA checkpoint.
    packed_dir = offload_dir / "packed-layers"
    if packed_dir.exists():
        shutil.rmtree(packed_dir)
    packed_dir.mkdir(parents=True, exist_ok=True)
    max_memory = {}
    if args.max_model_memory_gpu is not None or args.max_model_memory_cpu is not None:
        # When Accelerate receives a partial max_memory map, omitted devices
        # can be excluded from placement and cause weights to spill to disk.
        # Use its detected available capacities for unset tiers, then apply
        # explicit user caps (including an intentional 0-byte cap).
        from accelerate.utils import get_max_memory

        available = get_max_memory()
        max_memory[0] = (
            args.max_model_memory_gpu
            if args.max_model_memory_gpu is not None else available[0]
        )
        max_memory["cpu"] = (
            args.max_model_memory_cpu
            if args.max_model_memory_cpu is not None else available["cpu"]
        )
    log(
        f"Loading checkpoint with Accelerate offload; forward execution device is CUDA "
        f"(GPU 0), weights may be stored on GPU/CPU/disk; "
        f"placement limits={max_memory or 'unset (automatic)'}: {checkpoint}"
    )
    if LOG_LEVEL == "extensive":
        inventory_started = time.monotonic()
        checkpoint_bytes, checkpoint_files = checkpoint_weight_summary(checkpoint)
        checkpoint_disk = shutil.disk_usage(checkpoint)
        offload_disk = shutil.disk_usage(offload_dir)
        log(
            f"Checkpoint inventory: weight files={len(checkpoint_files)}, "
            f"total={checkpoint_bytes / 10**9:.2f} GB, "
            f"largest={[(name, round(size / 10**9, 2)) for name, size in checkpoint_files[:8]]}; "
            f"checkpoint filesystem free={checkpoint_disk.free / 10**9:.2f} GB, "
            f"offload filesystem free={offload_disk.free / 10**9:.2f} GB; "
            f"inventory scan={time.monotonic() - inventory_started:.2f}s",
            level="extensive",
        )
        if torch.cuda.is_available():
            free_cuda, total_cuda = torch.cuda.mem_get_info()
            log(
                f"Before checkpoint load: CUDA free/total={free_cuda / 10**9:.2f}/"
                f"{total_cuda / 10**9:.2f} GB; dtype=float16, device_map=auto, "
                f"offload_state_dict=True, attention={model_cfg.get('attn_implementation', 'sdpa')}",
                level="extensive",
            )
    model_load_started = time.monotonic()
    load_heartbeat_stop = threading.Event()

    def report_load_heartbeat() -> None:
        while not load_heartbeat_stop.wait(30):
            model_offload_dir = offload_dir / "model"
            offload_files = list(model_offload_dir.iterdir()) if model_offload_dir.exists() else []
            offload_bytes = sum(path.stat().st_size for path in offload_files if path.is_file())
            log(
                f"Checkpoint load still running; elapsed={time.monotonic() - model_load_started:.0f}s; "
                f"top-level offload files={len(offload_files)}, "
                f"offload bytes={offload_bytes / 10**9:.2f} GB; {cuda_memory_summary()}",
                level="extensive",
            )

    load_heartbeat = None
    if LOG_LEVEL == "extensive":
        load_heartbeat = threading.Thread(target=report_load_heartbeat, daemon=True)
        load_heartbeat.start()
    try:
        tokenizer, model, image_processor, _ = builder(
            str(checkpoint), None, "llava-v1.5-7b", device_map="auto",
            torch_dtype="float16", attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            max_memory=max_memory or None, offload_folder=str(offload_dir / "model"), offload_state_dict=True,
        )
    finally:
        load_heartbeat_stop.set()
        if load_heartbeat is not None:
            load_heartbeat.join()
        record_timing("LLaVA builder checkpoint load and dispatch", time.monotonic() - model_load_started)
        if LOG_LEVEL == "extensive":
            log(
                f"LLaVA builder call ended after {time.monotonic() - model_load_started:.1f}s; "
                f"{cuda_memory_summary()}",
                level="extensive",
            )
    model_eval_started = time.monotonic()
    model.eval()
    record_timing("model eval mode setup", time.monotonic() - model_eval_started)
    adapter_started = time.monotonic()
    adapter = get_process_model("llava")(model, tokenizer, image_processor)
    record_timing("multimodal adapter initialization", time.monotonic() - adapter_started)
    blocks = model.model.layers
    selected_layers = parse_layer_selection(args.layers, len(blocks))
    log(
        f"Quantizing {len(selected_layers)}/{len(blocks)} decoder layers: "
        f"{','.join(str(index + 1) for index in sorted(selected_layers))}"
    )
    device_map = getattr(model, "hf_device_map", {})
    log(
        f"Model has {len(blocks)} decoder layers; Accelerate storage map: "
        f"{summarize_device_map(device_map)}"
    )
    log(f"Full storage map: {device_map}", level="extensive")
    model_offload_dir = offload_dir / "model"
    model_offload_dir.mkdir(parents=True, exist_ok=True)
    model_disk_bytes = 0
    model_disk_file_count = 0
    for path in model_offload_dir.rglob("*"):
        if path.is_file():
            model_disk_file_count += 1
            model_disk_bytes += path.stat().st_size
    log(
        f"Model disk-offload cache: {model_disk_bytes / 1024**3:.2f} GiB currently present "
        f"under {model_offload_dir}; CPU and disk weights are staged to CUDA for forward execution"
    )
    if LOG_LEVEL == "extensive":
        offload_disk = shutil.disk_usage(model_offload_dir)
        log(
            f"After model dispatch: offload files={model_disk_file_count}, "
            f"offload bytes={model_disk_bytes / 10**9:.2f} GB, "
            f"offload filesystem free={offload_disk.free / 10**9:.2f} GB; {cuda_memory_summary()}",
            level="extensive",
        )
    linears_by_layer = [linear_layers(block) for block in blocks]
    for layer_index, linears in enumerate(linears_by_layer):
        if not linears:
            raise RuntimeError(f"Decoder layer {layer_index} contains no Linear modules")
    ram_budget_bytes = args.max_hessian_memory_cpu
    disk_budget_bytes = args.max_hessian_memory_disk
    hessian_allocation_started = time.monotonic()
    hessians, hessian_paths, hessian_placement, hessian_offsets, ram_bytes, disk_bytes = allocate_hessians(
        linears_by_layer, hessian_dir, ram_budget_bytes, disk_budget_bytes, selected_layers
    )
    record_timing("Hessian backing allocation", time.monotonic() - hessian_allocation_started)
    counts: dict[str, int] = {}  # Linear key -> selected activation rows accumulated across all groups' samples.
    token_selection: dict[str, torch.Tensor | None] = {"mask": None}  # CPU bool [batch, sequence] mask, or None.
    iga_state: dict[str, Any] = {
        "attention_mask": None,
        "vision_mask": None,
        "layer_weights": {},
        "pending_activations": {},
    }
    iga_cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    original_output_attentions = getattr(model.config, "output_attentions", False)
    if args.hessian_weighting == "iga":
        model.config.output_attentions = True
    # Probe after Accelerate materializes CPU/disk weights so the reported
    # parameter device reflects execution, rather than the pre-forward meta state.
    execution_probe_handles = []
    for layer_index, block in enumerate(blocks):
        def log_layer_execution(module, inputs, layer_index=layer_index):
            if getattr(module, "_gptq_execution_logged", False):
                return
            module._gptq_execution_logged = True
            input_device = (
                str(inputs[0].device)
                if inputs and torch.is_tensor(inputs[0])
                else "unavailable"
            )
            first_parameter = next(module.parameters(), None)
            weight_device = str(first_parameter.device) if first_parameter is not None else "none"
            hook = getattr(module, "_hf_hook", None)
            execution_device = getattr(hook, "execution_device", "not exposed")
            log(
                f"Forward execution probe decoder {layer_index + 1}/{len(blocks)}: "
                f"input_device={input_device}, parameter_device={weight_device}, "
                f"accelerate_execution_device={execution_device}",
                level="extensive",
            )

        execution_probe_handles.append(register_after_weights_loaded(block, log_layer_execution))

    # Keep one group's exact Hessians on CUDA for its complete calibration pass.
    # This trades repeated model forwards for one disk save per group instead
    # of loading and flushing every group for every sample.
    collection_started = time.monotonic()
    group_specs = []
    for group_start in range(0, len(blocks), args.hessian_group_layers):
        group_end = min(group_start + args.hessian_group_layers, len(blocks))
        group_layers = [
            index for index in range(group_start, group_end) if index in selected_layers
        ]
        if not group_layers:
            continue
        group_keys = [
            f"{layer_index}:{name}"
            for layer_index in group_layers
            for name in linears_by_layer[layer_index]
        ]
        group_bytes = sum(hessians[key].nbytes for key in group_keys)
        if args.max_hessian_memory_gpu is not None and group_bytes > args.max_hessian_memory_gpu:
            raise RuntimeError(
                f"GPU Hessian group {group_start + 1}-{group_end} needs "
                f"{format_memory_gb(group_bytes)}, exceeding "
                f"--max-hessian-memory-gpu={format_memory_gb(args.max_hessian_memory_gpu)}; "
                "reduce --hessian-group-layers"
            )
        disk_layout = [hessian_offsets.get(key) for key in group_keys]
        disk_chunk = None
        if disk_layout and all(item is not None for item in disk_layout):
            offset = disk_layout[0][0]
            expected = offset
            for item_offset, elements in disk_layout:
                if item_offset != expected:
                    break
                expected += elements * np.dtype(np.float32).itemsize
            else:
                disk_chunk = (offset, sum(item[1] for item in disk_layout))
        group_specs.append((group_layers, group_keys, group_bytes, disk_chunk))

    total_forwards = len(dataset) * len(group_specs)
    log(
        f"Collecting exact Hessians in {len(group_specs)} group passes; "
        f"{len(dataset)} samples/group, {total_forwards} total model forwards, "
        f"{disk_bytes / 10**9:.2f} GB Hessians disk-backed and written in "
        f"at most {len(group_specs)} group saves, up to "
        f"{args.hessian_group_layers} selected decoder layers resident on GPU"
    )
    completed_forwards = 0
    active_hessians: dict[str, torch.Tensor] = {}
    active_chunk: torch.Tensor | None = None
    try:
        for group_pass, (group_layers, group_keys, group_bytes, disk_chunk) in enumerate(group_specs, 1):
            group_start, group_end = min(group_layers), max(group_layers) + 1
            if active_hessians:
                raise RuntimeError("A Hessian group is still active before the next group pass")

            if LOG_LEVEL == "extensive":
                torch.cuda.synchronize()
            load_started = time.monotonic()
            if disk_chunk is not None:
                offset, elements = disk_chunk
                disk_view = np.memmap(
                    hessian_paths[group_keys[0]], mode="r+", dtype=np.float32,
                    offset=offset, shape=(elements,),
                )
                active_chunk = torch.from_numpy(disk_view).to(device="cuda")
                cursor = 0
                for key in group_keys:
                    matrix = hessians[key]
                    matrix_elements = matrix.size
                    active_hessians[key] = active_chunk[cursor:cursor + matrix_elements].view(matrix.shape)
                    cursor += matrix_elements
                del disk_view
            else:
                for key in group_keys:
                    host_matrix = np.array(hessians[key], copy=True)
                    active_hessians[key] = torch.from_numpy(host_matrix).to(device="cuda")
            if LOG_LEVEL == "extensive":
                torch.cuda.synchronize()
            load_elapsed = time.monotonic() - load_started
            record_timing("Hessian group backing-to-GPU load", load_elapsed)
            log(
                f"Group pass {group_pass}/{len(group_specs)}: layers="
                f"{','.join(str(index + 1) for index in group_layers)}, Hessian="
                f"{group_bytes / 1024**3:.2f} GiB loaded in {load_elapsed:.1f}s; "
                f"{cuda_memory_summary()}",
                level="extensive",
            )

            collect_handles = []
            if args.hessian_weighting == "uniform":
                for layer_index in group_layers:
                    for name, linear in linears_by_layer[layer_index].items():
                        key = f"{layer_index}:{name}"
                        collect_handles.append(linear.register_forward_hook(
                            gpu_hessian_hook(active_hessians, linear.in_features, counts, key, token_selection)
                        ))
            else:
                active_layer_set = set(group_layers)
                for layer_index in group_layers:
                    for name, linear in linears_by_layer[layer_index].items():
                        key = f"{layer_index}:{name}"

                        def capture_iga_activation(
                            _module,
                            inputs,
                            _output,
                            *,
                            layer_index=layer_index,
                            name=name,
                            key=key,
                            width=linear.in_features,
                        ):
                            if not inputs or not torch.is_tensor(inputs[0]):
                                raise RuntimeError(f"Could not capture activation rows for IGA Hessian {key}")
                            activations = inputs[0].detach()
                            if name.startswith("self_attn."):
                                # IGA is returned by self_attn after q/k/v/o
                                # projections execute, so defer these four X'WX updates.
                                iga_state["pending_activations"][key] = activations
                                return
                            token_weights = iga_state["layer_weights"].get(layer_index)
                            if token_weights is None:
                                raise RuntimeError(f"IGA scores are unavailable before MLP activation {key}")
                            started_event = torch.cuda.Event(enable_timing=True) if SHOW_TIMING else None
                            if started_event is not None:
                                started_event.record()
                            accumulate_weighted_hessian(
                                active_hessians,
                                counts,
                                key,
                                width,
                                activations,
                                iga_state["attention_mask"],
                                token_weights,
                            )
                            if started_event is not None:
                                ended_event = torch.cuda.Event(enable_timing=True)
                                ended_event.record()
                                iga_cuda_events.append(("IGA weighted Hessian accumulation", started_event, ended_event))

                        collect_handles.append(linear.register_forward_hook(capture_iga_activation))

                for layer_index, block in enumerate(blocks):
                    attention_module = getattr(block, "self_attn", None)
                    if attention_module is None:
                        raise RuntimeError(f"Decoder layer {layer_index + 1} has no self_attn module for IGA")

                    def capture_layer_iga(_module, _inputs, output, *, layer_index=layer_index):
                        if not isinstance(output, tuple) or len(output) < 2:
                            if layer_index in active_layer_set:
                                raise RuntimeError(
                                    f"Decoder layer {layer_index + 1} did not return attention weights for IGA; "
                                    "check that the model uses eager attention"
                                )
                            return
                        attention_weights = output[1]
                        if layer_index in active_layer_set:
                            if not torch.is_tensor(attention_weights):
                                raise RuntimeError(
                                    f"Decoder layer {layer_index + 1} returned no attention weights for IGA; "
                                    "check that the model uses eager attention"
                                )
                            iga_started_event = torch.cuda.Event(enable_timing=True) if SHOW_TIMING else None
                            if iga_started_event is not None:
                                iga_started_event.record()
                            token_weights = make_iga_token_weights(
                                attention_weights,
                                iga_state["attention_mask"],
                                iga_state["vision_mask"],
                            )
                            if iga_started_event is not None:
                                iga_ended_event = torch.cuda.Event(enable_timing=True)
                                iga_ended_event.record()
                                iga_cuda_events.append(("IGA attention score calculation", iga_started_event, iga_ended_event))
                            iga_state["layer_weights"][layer_index] = token_weights
                            for name, linear in linears_by_layer[layer_index].items():
                                if not name.startswith("self_attn."):
                                    continue
                                key = f"{layer_index}:{name}"
                                activations = iga_state["pending_activations"].pop(key, None)
                                if activations is None:
                                    raise RuntimeError(f"No saved attention-projection activations for IGA Hessian {key}")
                                started_event = torch.cuda.Event(enable_timing=True) if SHOW_TIMING else None
                                if started_event is not None:
                                    started_event.record()
                                accumulate_weighted_hessian(
                                    active_hessians,
                                    counts,
                                    key,
                                    linear.in_features,
                                    activations,
                                    iga_state["attention_mask"],
                                    token_weights,
                                )
                                if started_event is not None:
                                    ended_event = torch.cuda.Event(enable_timing=True)
                                    ended_event.record()
                                    iga_cuda_events.append(("IGA weighted Hessian accumulation", started_event, ended_event))
                        # Prevent the model output from retaining a full attention
                        # matrix for every decoder layer.
                        if torch.is_tensor(attention_weights):
                            return (output[0], None, *output[2:])

                    collect_handles.append(attention_module.register_forward_hook(capture_layer_iga))

            group_started = time.monotonic()
            try:
                for sample_index, sample in enumerate(dataset):
                    sample_started = time.monotonic()
                    preparation_started = time.monotonic()
                    image_value = sample["image"]
                    if isinstance(image_value, dict):
                        image_value = Image.open(image_value["path"]).convert("RGB")
                    elif hasattr(image_value, "convert"):
                        image_value = image_value.convert("RGB")
                    row = {
                        "conversations": sample["conversations"],
                        "id": str(sample.get("id", sample_index)),
                        "image": "local",
                    }
                    prepared = adapter.preprocess_data([image_value], row)
                    batch = adapter.data_collator([prepared])
                    inputs, kwargs = adapter.generate_input(batch)
                    preparation_elapsed = time.monotonic() - preparation_started
                    record_timing("calibration sample preprocessing", preparation_elapsed)
                    vision_mask = kwargs["vision_mask"].detach().to(device="cpu", dtype=torch.bool)
                    attention_mask = kwargs["attention_mask"].detach().to(device="cpu", dtype=torch.bool)
                    if args.hessian_weighting == "iga":
                        iga_state["attention_mask"] = attention_mask[0].to(device="cuda")
                        iga_state["vision_mask"] = vision_mask[0].to(device="cuda")
                        image_positions = attention_mask[0] & vision_mask[0]
                        if not image_positions.any():
                            raise RuntimeError("IGA weighting requires at least one valid image token")
                        image_end = image_positions.nonzero(as_tuple=True)[0][-1]
                        text_queries = (
                            attention_mask[0]
                            & ~vision_mask[0]
                            & (torch.arange(attention_mask.shape[-1]) > image_end)
                        )
                        if not text_queries.any():
                            raise RuntimeError("IGA weighting requires a valid text query after the image tokens")
                        iga_state["layer_weights"].clear()
                        iga_state["pending_activations"].clear()
                        iga_cuda_events.clear()
                    token_selection["mask"] = (
                        attention_mask & ~vision_mask if args.hessian_token_scope == "text" else None
                    )
                    valid_tokens = int(attention_mask.sum().item())
                    vision_tokens = int((attention_mask & vision_mask).sum().item())
                    selected_tokens = (
                        int((attention_mask & ~vision_mask).sum().item())
                        if args.hessian_token_scope == "text" else valid_tokens
                    )
                    log(
                        f"Group pass {group_pass}/{len(group_specs)}, sample "
                        f"{sample_index + 1}/{len(dataset)} prepared in {preparation_elapsed:.1f}s; "
                        f"sequence={attention_mask.shape[-1]}, valid={valid_tokens}, "
                        f"image={vision_tokens}, selected={selected_tokens} tokens"
                    )
                    forward_started = time.monotonic()
                    try:
                        with torch.inference_mode():
                            model(
                                inputs_embeds=inputs["inputs_embeds"],
                                attention_mask=kwargs.get("attention_mask"),
                                use_cache=False,
                                output_attentions=args.hessian_weighting == "iga",
                                return_dict=True,
                            )
                        torch.cuda.synchronize()
                    finally:
                        token_selection["mask"] = None
                    for stage, started_event, ended_event in iga_cuda_events:
                        record_timing(stage, started_event.elapsed_time(ended_event) / 1000.0)
                    if args.hessian_weighting == "iga" and iga_state["pending_activations"]:
                        missing = ", ".join(iga_state["pending_activations"])
                        iga_state["pending_activations"].clear()
                        raise RuntimeError(f"IGA attention hooks did not consume captured activations: {missing}")
                    forward_elapsed = time.monotonic() - forward_started
                    completed_forwards += 1
                    record_timing("calibration model forward with active Hessian group", forward_elapsed)
                    del inputs, kwargs, batch, prepared, image_value
                    elapsed = time.monotonic() - collection_started
                    average = elapsed / completed_forwards
                    remaining = average * (total_forwards - completed_forwards)
                    log(
                        f"Group pass {group_pass}/{len(group_specs)}, sample "
                        f"{sample_index + 1}/{len(dataset)} complete; forward={forward_elapsed:.1f}s, "
                        f"total={time.monotonic() - sample_started:.1f}s, "
                        f"completed forwards={completed_forwards}/{total_forwards}, "
                        f"estimated remaining={remaining / 60:.1f} min; {cuda_memory_summary()}"
                    )
            finally:
                token_selection["mask"] = None
                iga_state["layer_weights"].clear()
                iga_state["pending_activations"].clear()
                for handle in collect_handles:
                    handle.remove()

            if active_chunk is not None:
                torch.cuda.synchronize()
                save_started = time.monotonic()
                copy_started = time.monotonic()
                cpu_chunk = active_chunk.cpu()
                copy_elapsed = time.monotonic() - copy_started
                offset, elements = disk_chunk
                disk_view = np.memmap(
                    hessian_paths[group_keys[0]], mode="r+", dtype=np.float32,
                    offset=offset, shape=(elements,),
                )
                assignment_started = time.monotonic()
                disk_view[:] = cpu_chunk.numpy()
                assignment_elapsed = time.monotonic() - assignment_started
                flush_started = time.monotonic()
                disk_view.flush()
                flush_elapsed = time.monotonic() - flush_started
                del disk_view, cpu_chunk
                active_chunk = None
            else:
                save_started = time.monotonic()
                copy_elapsed = 0.0
                assignment_elapsed = 0.0
                flush_elapsed = 0.0
                for key in group_keys:
                    copy_started = time.monotonic()
                    cpu_matrix = active_hessians[key].cpu()
                    copy_elapsed += time.monotonic() - copy_started
                    assignment_started = time.monotonic()
                    hessians[key][:] = cpu_matrix.numpy()
                    assignment_elapsed += time.monotonic() - assignment_started
                    if isinstance(hessians[key], np.memmap):
                        flush_started = time.monotonic()
                        hessians[key].flush()
                        flush_elapsed += time.monotonic() - flush_started
                    del cpu_matrix
            write_elapsed = assignment_elapsed + flush_elapsed
            active_hessians.clear()
            save_elapsed = time.monotonic() - save_started
            record_timing("Hessian group save total", save_elapsed)
            record_timing("Hessian GPU-to-host copy", copy_elapsed)
            record_timing("Hessian memmap assignment", assignment_elapsed)
            record_timing("Hessian memmap flush", flush_elapsed)
            log(
                f"Group pass {group_pass}/{len(group_specs)} completed for layers "
                f"{','.join(str(index + 1) for index in group_layers)} in "
                f"{time.monotonic() - group_started:.1f}s; saved Hessians once: "
                f"GPU-to-host={copy_elapsed:.1f}s, memmap assignment={assignment_elapsed:.1f}s, "
                f"flush={flush_elapsed:.1f}s, write/flush total={write_elapsed:.1f}s, "
                f"total={save_elapsed:.1f}s; {cuda_memory_summary()}"
            )
    finally:
        token_selection["mask"] = None
        active_hessians.clear()
        active_chunk = None
        if args.hessian_weighting == "iga":
            model.config.output_attentions = original_output_attentions
    for handle in execution_probe_handles:
        handle.remove()
    token_counts = list(counts.values())
    log(
        f"Hessian collection finished in {time.monotonic() - collection_started:.1f}s; "
        f"captured {sum(token_counts):,} activation rows across {len(counts):,} Linear modules "
        f"(per-module min/median/max={min(token_counts, default=0):,}/"
        f"{int(np.median(token_counts)) if token_counts else 0:,}/{max(token_counts, default=0):,})"
    )
    if args.hessian_token_scope == "text":
        empty_linears = [key for key in hessians if counts.get(key, 0) == 0]
        if empty_linears:
            raise RuntimeError(f"No text-token activations were collected for: {', '.join(empty_linears)}")

    quantized_layers: set[int] = set()
    quantize_handles = []
    for layer_index, (block, linears) in enumerate(zip(blocks, linears_by_layer)):
        if layer_index not in selected_layers:
            continue
        packed_path = packed_dir / f"decoder-{layer_index:03d}.pt"

        def quantize_layer(
            _module,
            _args,
            layer_index=layer_index,
            linears=linears,
            packed_path=packed_path,
        ):
            layer_started = time.monotonic()
            log(f"Starting decoder layer {layer_index + 1}/{len(blocks)} ({len(linears)} Linear modules)")
            state = {}
            for name, linear in linears.items():
                key = f"{layer_index}:{name}"
                linear_started = time.monotonic()
                state[name] = quantize_linear_gptq(
                    linear,
                    hessians[key],
                    counts.get(key, 0),
                    bits=args.bits,
                    group_size=args.group_size,
                    damp=args.damp,
                )
                linear_elapsed = time.monotonic() - linear_started
                record_timing("GPTQ Linear quantization", linear_elapsed)
                log(
                    f"Quantized decoder {layer_index + 1}/{len(blocks)} {name}: "
                    f"shape={tuple(linear.weight.shape)}, Hessian tokens={counts.get(key, 0):,}, "
                    f"storage={hessian_placement[key]}, elapsed={linear_elapsed:.1f}s"
                )
                release_hessian(key, hessians, hessian_paths, args.keep_hessians)
            temporary = packed_path.with_suffix(".tmp")
            save_started = time.monotonic()
            torch.save(state, temporary)
            os.replace(temporary, packed_path)
            record_timing("quantized layer serialization and save", time.monotonic() - save_started)
            quantized_layers.add(layer_index)
            log(
                f"Saved {packed_path.name} ({packed_path.stat().st_size / 1024**2:.1f} MiB); "
                f"decoder layer elapsed={time.monotonic() - layer_started:.1f}s, "
                f"completed={len(quantized_layers)}/{len(selected_layers)}"
            )

        quantize_handles.append(register_after_weights_loaded(block, quantize_layer))

    # This forward loads each selected offloaded decoder layer. Its callback
    # runs after Accelerate materializes that layer, then quantizes and saves it.
    # The Hessians above all came from base-model forwards.
    first_sample = dataset[0]
    finalization_preparation_started = time.monotonic()
    image_value = first_sample["image"]
    if isinstance(image_value, dict):
        image_value = Image.open(image_value["path"]).convert("RGB")
    elif hasattr(image_value, "convert"):
        image_value = image_value.convert("RGB")
    row = {
        "conversations": first_sample["conversations"],
        "id": str(first_sample.get("id", 0)),
        "image": "local",
    }
    prepared = adapter.preprocess_data([image_value], row)
    batch = adapter.data_collator([prepared])
    inputs, kwargs = adapter.generate_input(batch)
    record_timing("finalization sample preprocessing", time.monotonic() - finalization_preparation_started)
    finalization_started = time.monotonic()
    try:
        with torch.inference_mode():
            model(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=kwargs.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
    finally:
        record_timing("finalization forward and layer quantization hooks", time.monotonic() - finalization_started)
        for handle in quantize_handles:
            handle.remove()
    del inputs, kwargs, batch, prepared, image_value
    if len(quantized_layers) != len(selected_layers):
        raise RuntimeError(
            f"Quantized {len(quantized_layers)} of {len(selected_layers)} selected decoder layers during finalization"
        )

    # Release the offloaded calibration model before loading a complete CPU
    # copy for serialization. The standalone checkpoint keeps the base model's
    # unselected layers and replaces selected Linear weights with GPTQ values.
    quantize_handles.clear()
    execution_probe_handles.clear()
    # The loop's final callback and loop variables retain its decoder block.
    # Drop them as well so the offloaded calibration model can be released
    # before the complete CPU model is loaded for serialization.
    del quantize_layer, log_layer_execution, block, linears, linear
    if args.hessian_weighting == "iga":
        del capture_iga_activation, capture_layer_iga, attention_module
    del adapter, model, blocks, linears_by_layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    export_started = time.monotonic()
    log(f"Loading base checkpoint on CPU to write standalone model: {checkpoint}")
    tokenizer, export_model, image_processor, _ = builder(
        str(checkpoint), None, "llava-v1.5-7b", device_map="cpu",
        torch_dtype="float16", attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    # This builder moves the vision tower to CUDA for non-auto device maps.
    # Return the complete model to CPU before applying weights and serializing.
    export_model.to("cpu")
    export_model.eval()
    export_blocks = export_model.model.layers
    apply_started = time.monotonic()
    applied_layers = 0
    for packed_path in sorted(packed_dir.glob("decoder-*.pt")):
        layer_index = int(packed_path.stem.removeprefix("decoder-"))
        if not 0 <= layer_index < len(export_blocks):
            raise ValueError(f"Overlay layer {layer_index} is outside model with {len(export_blocks)} layers")
        state = torch.load(packed_path, map_location="cpu", weights_only=True)
        export_linears = linear_layers(export_blocks[layer_index])
        for name, entry in state.items():
            if name not in export_linears:
                raise ValueError(f"Overlay Linear {name!r} not found in decoder layer {layer_index + 1}")
            parameter = export_linears[name].weight
            dense_weight = dequantize_entry(entry, torch.device("cpu"), parameter.dtype)
            if dense_weight.shape != parameter.shape:
                raise ValueError(
                    f"Shape mismatch for decoder {layer_index + 1} {name}: "
                    f"packed={tuple(dense_weight.shape)}, model={tuple(parameter.shape)}"
                )
            parameter.data.copy_(dense_weight)
        applied_layers += 1
        if LOG_LEVEL == "extensive":
            log(f"Applied quantized weights for decoder {layer_index + 1}/{len(export_blocks)}")
    if applied_layers != len(selected_layers):
        raise RuntimeError(
            f"Applied {applied_layers} packed decoder layers; expected {len(selected_layers)}"
        )
    record_timing("standalone model packed-layer application", time.monotonic() - apply_started)

    serialization_started = time.monotonic()
    # Match the configured LLaVA base checkpoint format: PyTorch .bin shards.
    # This keeps the standalone output directly usable by the existing LLaVA
    # builder and makes the serialized format explicit instead of relying on
    # Transformers' safetensors default.
    export_model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)
    if image_processor is not None:
        image_processor.save_pretrained(output_dir)
    output_weight_files = [
        path for path in output_dir.glob("pytorch_model*.bin") if path.is_file()
    ]
    if not (output_dir / "config.json").is_file() or not output_weight_files:
        raise RuntimeError(
            "Standalone checkpoint export is incomplete: expected config.json and "
            f"PyTorch weight shards under {output_dir}"
        )
    serialization_elapsed = time.monotonic() - serialization_started
    record_timing("standalone model serialization and save", serialization_elapsed)
    if LOG_LEVEL == "extensive":
        saved_weight_bytes, saved_weight_files = checkpoint_weight_summary(output_dir)
        throughput = saved_weight_bytes / max(serialization_elapsed, 1e-9) / 10**6
        log(
            f"Standalone checkpoint saved: weight files={len(saved_weight_files)}, "
            f"weight bytes={saved_weight_bytes / 10**9:.2f} GB, "
            f"serialization throughput={throughput:.1f} MB/s",
            level="extensive",
        )
    # export_blocks and the final loop's tensors are references into the model;
    # release them before collecting so the full CPU checkpoint is freed.
    del export_blocks, export_linears, parameter, dense_weight, state, packed_path
    del export_model
    gc.collect()
    record_timing("standalone model export", time.monotonic() - export_started)

    metadata = {
        "format": "attention-quantization-gptq-standalone-pytorch-bin-dense-v1",
        "base_model": str(checkpoint),
        "model_family": "llava-v1.5",
        "bits": args.bits,
        "group_size": args.group_size,
        "selected_layers_1based": sorted(index + 1 for index in selected_layers),
        "hessian_token_scope": args.hessian_token_scope,
        "hessian_weighting": args.hessian_weighting,
        "hessian_memory_limits_bytes": {
            "gpu_bytes": args.max_hessian_memory_gpu,
            "cpu_bytes": args.max_hessian_memory_cpu,
            "disk_bytes": args.max_hessian_memory_disk,
        },
        "hessian_cpu_used_gib": ram_bytes / (1024**3),
        "hessian_disk_used_gib": disk_bytes / (1024**3),
        "hessian_disk_files": (
            sorted({str(path.relative_to(offload_dir)) for path in hessian_paths.values()})
            if args.keep_hessians else []
        ),
        "hessian_placement": hessian_placement,
        "damp": args.damp,
        "calibration_samples": len(dataset),
        "seed": args.seed,
        "storage": "dense float16 standalone checkpoint in PyTorch .bin format",
        "load_note": "Standalone LLaVA checkpoint; load directly from this output directory.",
    }
    (output_dir / "gptq_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    cleanup_started = time.monotonic()
    shutil.rmtree(packed_dir, ignore_errors=True)
    if not args.keep_hessians and hessian_dir.exists():
        shutil.rmtree(hessian_dir)
    shutil.rmtree(offload_dir / "model", ignore_errors=True)
    if args.offload_dir is None and not args.keep_hessians:
        shutil.rmtree(offload_dir, ignore_errors=True)
    record_timing("temporary Hessian/model scratch cleanup", time.monotonic() - cleanup_started)
    log(f"Saved standalone dense FP16 GPTQ model: {output_dir}")
    if SHOW_TIMING and LOG_LEVEL != "extensive":
        log(f"Run completed in {time.monotonic() - run_started:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
