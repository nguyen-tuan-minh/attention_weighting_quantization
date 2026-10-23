"""Disk-backed, layerwise GPTQ for the project's LLaVA 1.5 checkpoint.

The output is an overlay of quantized decoder Linear weights. It does not
modify LMMS-Eval or rewrite/copy the base checkpoint. Use ``apply_overlay``
from this module after loading the base model to run the quantized model.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
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


def log(message: str) -> None:
    print(f"[gptq] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the quantized weight overlay.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset.yaml")
    parser.add_argument("--base-model", default=None, help="Override model.yaml's local checkpoint path.")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--hessian-token-scope", choices=("all", "text"), default="all",
        help="Accumulate Hessians from all valid tokens or only non-image text tokens.",
    )
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--max-memory-gpu", default="10GiB")
    parser.add_argument("--max-memory-cpu", default="2GiB")
    parser.add_argument(
        "--hessian-ram-gib", type=float, default=None,
        help="RAM budget for Hessian matrices; default None means unlimited, remaining matrices are disk-backed.",
    )
    parser.add_argument("--offload-dir", type=Path, default=None, help="Scratch disk directory; defaults under output-dir.")
    parser.add_argument("--keep-hessians", action="store_true", help="Keep Hessian files after success for inspection/resume work.")
    parser.add_argument("--overwrite", action="store_true")
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
    configured = args.base_model or model_config.get("model_dir")
    if not configured:
        raise ValueError("Set model_dir in model.yaml or pass --base-model")
    checkpoint = repository_path(configured)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {checkpoint / 'config.json'}")
    return checkpoint


def sample_records(args: argparse.Namespace, data_config: dict[str, Any], dataset_config_path: Path):
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
    return {name: module for name, module in block.named_modules() if isinstance(module, nn.Linear)}


def hessian_hook(
    hessian: np.ndarray,
    width: int,
    tile: int,
    counts: dict[str, int],
    key: str,
    token_selection: dict[str, torch.Tensor | None],
):
    """Accumulate X'X into a RAM array or disk memmap using bounded GPU tiles."""
    def capture(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0].detach().reshape(-1, width).float()
        # mask: bool[batch, sequence], selecting only token rows used in H.
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
        x = x.to(device="cuda", dtype=torch.float32)
        for row_start in range(0, width, tile):
            row_end = min(row_start + tile, width)
            left = x[:, row_start:row_end]
            for col_start in range(row_start, width, tile):
                col_end = min(col_start + tile, width)
                gram = (left.T @ x[:, col_start:col_end]).cpu().numpy()
                hessian[row_start:row_end, col_start:col_end] += gram
                if col_start != row_start:
                    hessian[col_start:col_end, row_start:row_end] += gram.T
        del x

    return capture


def allocate_hessians(
    blocks: Any,
    hessian_dir: Path,
    ram_budget_bytes: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Path], dict[str, str], int, int]:
    """Allocate Hessian matrices greedily in RAM, then as disk memmaps."""
    linears_by_layer = [linear_layers(block) for block in blocks]
    entries = [
        (f"{layer_index}:{name}", linear)
        for layer_index, linears in enumerate(linears_by_layer)
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

    hessians: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    placement: dict[str, str] = {}
    for key, linear, _matrix_bytes, in_ram in planned:
        if in_ram:
            hessians[key] = np.zeros((linear.in_features, linear.in_features), dtype=np.float32)
            placement[key] = "ram"
            continue
        layer_index, name = key.split(":", 1)
        path = hessian_dir / f"layer-{int(layer_index):03d}-{name.replace('.', '_')}.f32"
        path.unlink(missing_ok=True)
        hessians[key] = np.memmap(
            path, mode="w+", dtype=np.float32,
            shape=(linear.in_features, linear.in_features),
        )
        paths[key] = path
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
    return hessians, paths, placement, ram_bytes, disk_bytes


def flush_disk_hessians(hessians: dict[str, np.ndarray]) -> None:
    """Flush all disk-backed Hessian arrays after calibration collection."""
    for hessian in hessians.values():
        if isinstance(hessian, np.memmap):
            hessian.flush()


def release_hessian(key: str, hessians: dict[str, np.ndarray], paths: dict[str, Path], keep: bool) -> None:
    """Release one matrix after it has been quantized; optionally delete its file."""
    hessian = hessians.pop(key, None)
    if isinstance(hessian, np.memmap):
        hessian.flush()
        hessian._mmap.close()
    path = paths.get(key)
    if path is not None and not keep:
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


def apply_overlay(model: nn.Module, overlay_dir: str | Path) -> list[Any]:
    """Attach pre-forward hooks that reapply disk-saved quantized layers."""
    overlay_dir = Path(overlay_dir)
    handles = []
    layers = model.model.layers
    for index, block in enumerate(layers):
        overlay_path = overlay_dir / f"decoder-{index:03d}.pt"
        if not overlay_path.exists():
            continue
        state = torch.load(overlay_path, map_location="cpu", weights_only=True)
        linears = linear_layers(block)
        def restore(_module: nn.Module, _args: tuple[Any, ...], state=state, linears=linears):
            for name, entry in state.items():
                parameter = linears[name].weight
                parameter.data.copy_(dequantize_entry(entry, parameter.device, parameter.dtype))
        handles.append(block.register_forward_pre_hook(restore))
    return handles


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.group_size < 1:
        raise ValueError("--group-size must be positive")
    if args.hessian_ram_gib is not None and args.hessian_ram_gib < 0:
        raise ValueError("--hessian-ram-gib must be nonnegative")
    if args.damp < 0:
        raise ValueError("--damp must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA for layer execution and Hessian factorization")
    output_dir = repository_path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = read_yaml(args.model_config)
    data_cfg = read_yaml(args.dataset_config)
    checkpoint = resolve_checkpoint(args, model_cfg)
    qig_root = repository_path(model_cfg.get("implementation_source_dir", "third_party/QIG"))
    builder, get_process_model = load_runtime(qig_root)
    offload_dir = repository_path(args.offload_dir) if args.offload_dir else output_dir / ".gptq-work"
    offload_dir.mkdir(parents=True, exist_ok=True)
    cleanup_default_scratch = args.offload_dir is None and not args.keep_hessians
    if cleanup_default_scratch:
        atexit.register(shutil.rmtree, offload_dir, ignore_errors=True)
    hessian_dir = offload_dir / "hessians"
    hessian_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "layers"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dataset = sample_records(args, data_cfg, args.dataset_config)
    max_memory = {0: args.max_memory_gpu, "cpu": args.max_memory_cpu}
    log(f"Loading checkpoint with automatic GPU/CPU/disk placement: {checkpoint}")
    tokenizer, model, image_processor, _ = builder(
        str(checkpoint), None, "llava-v1.5-7b", device_map="auto",
        torch_dtype="float16", attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        max_memory=max_memory, offload_folder=str(offload_dir / "model"), offload_state_dict=True,
    )
    model.eval()
    adapter = get_process_model("llava")(model, tokenizer, image_processor)
    blocks = model.model.layers
    log(f"Model has {len(blocks)} decoder layers; device map: {getattr(model, 'hf_device_map', {})}")
    linears_by_layer = [linear_layers(block) for block in blocks]
    for layer_index, linears in enumerate(linears_by_layer):
        if not linears:
            raise RuntimeError(f"Decoder layer {layer_index} contains no Linear modules")
    ram_budget_bytes = (
        None if args.hessian_ram_gib is None else int(args.hessian_ram_gib * (1024**3))
    )
    hessians, hessian_paths, hessian_placement, ram_bytes, disk_bytes = allocate_hessians(
        blocks, hessian_dir, ram_budget_bytes
    )
    counts: dict[str, int] = {}
    token_selection: dict[str, torch.Tensor | None] = {"mask": None}
    collect_handles = []
    for layer_index, linears in enumerate(linears_by_layer):
        for name, linear in linears.items():
            key = f"{layer_index}:{name}"
            collect_handles.append(
                linear.register_forward_hook(
                    hessian_hook(
                        hessians[key], linear.in_features, 1024, counts, key, token_selection
                    )
                )
            )

    # One unquantized full-model forward per sample accumulates every layer's
    # Hessian online. Only Hessian matrices are retained; token activations are
    # consumed by the hooks and discarded immediately.
    log(f"Collecting original-model Hessians from {len(dataset)} samples")
    try:
        for sample_index, sample in enumerate(dataset):
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
            # mask: bool[batch, sequence], optionally selecting text tokens only.
            vision_mask = kwargs["vision_mask"].detach().to(device="cpu", dtype=torch.bool)
            attention_mask = kwargs["attention_mask"].detach().to(device="cpu", dtype=torch.bool)
            token_selection["mask"] = (
                attention_mask & ~vision_mask if args.hessian_token_scope == "text" else None
            )
            try:
                with torch.inference_mode():
                    model(
                        inputs_embeds=inputs["inputs_embeds"],
                        attention_mask=kwargs.get("attention_mask"),
                        use_cache=False,
                        return_dict=True,
                    )
            finally:
                token_selection["mask"] = None
            del inputs, kwargs, batch, prepared, image_value
            log(f"Accumulated Hessians from sample {sample_index + 1}/{len(dataset)}")
    finally:
        token_selection["mask"] = None
        for handle in collect_handles:
            handle.remove()
    flush_disk_hessians(hessians)
    if args.hessian_token_scope == "text":
        empty_linears = [key for key in hessians if counts.get(key, 0) == 0]
        if empty_linears:
            raise RuntimeError(f"No text-token activations were collected for: {', '.join(empty_linears)}")

    quantized_layers: set[int] = set()
    quantize_handles = []
    for layer_index, (block, linears) in enumerate(zip(blocks, linears_by_layer)):
        overlay_path = overlay_dir / f"decoder-{layer_index:03d}.pt"

        def quantize_layer(
            _module,
            _args,
            layer_index=layer_index,
            linears=linears,
            overlay_path=overlay_path,
        ):
            state = {}
            for name, linear in linears.items():
                key = f"{layer_index}:{name}"
                state[name] = quantize_linear_gptq(
                    linear,
                    hessians[key],
                    counts.get(key, 0),
                    bits=args.bits,
                    group_size=args.group_size,
                    damp=args.damp,
                )
                log(f"Quantized layer {layer_index + 1}/{len(blocks)}: {name}")
                release_hessian(key, hessians, hessian_paths, args.keep_hessians)
            temporary = overlay_path.with_suffix(".tmp")
            torch.save(state, temporary)
            os.replace(temporary, overlay_path)
            quantized_layers.add(layer_index)

        quantize_handles.append(block.register_forward_pre_hook(quantize_layer))

    # This one forward loads each offloaded decoder layer, allowing its hook to
    # quantize and save it. The Hessians above all came from base-model forwards.
    first_sample = dataset[0]
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
    try:
        with torch.inference_mode():
            model(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=kwargs.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in quantize_handles:
            handle.remove()
    del inputs, kwargs, batch, prepared, image_value
    if len(quantized_layers) != len(blocks):
        raise RuntimeError(
            f"Quantized {len(quantized_layers)} of {len(blocks)} decoder layers during finalization"
        )

    metadata = {
        "format": "attention-quantization-gptq-packed-overlay-v1",
        "base_model": str(checkpoint),
        "model_family": "llava-v1.5",
        "bits": args.bits,
        "group_size": args.group_size,
        "hessian_token_scope": args.hessian_token_scope,
        "hessian_ram_budget_gib": args.hessian_ram_gib,
        "hessian_ram_used_gib": ram_bytes / (1024**3),
        "hessian_disk_used_gib": disk_bytes / (1024**3),
        "hessian_disk_files": (
            [str(path.relative_to(offload_dir)) for path in hessian_paths.values()]
            if args.keep_hessians else []
        ),
        "hessian_placement": hessian_placement,
        "damp": args.damp,
        "calibration_samples": len(dataset),
        "seed": args.seed,
        "overlay_files": [f"layers/decoder-{i:03d}.pt" for i in range(len(blocks))],
        "load_note": "Load the base model with the same architecture, then call apply_overlay(model, output_dir / 'layers').",
    }
    (output_dir / "gptq_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.keep_hessians and hessian_dir.exists():
        shutil.rmtree(hessian_dir)
    shutil.rmtree(offload_dir / "model", ignore_errors=True)
    if args.offload_dir is None and not args.keep_hessians:
        shutil.rmtree(offload_dir, ignore_errors=True)
    log(f"Saved GPTQ overlay: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
