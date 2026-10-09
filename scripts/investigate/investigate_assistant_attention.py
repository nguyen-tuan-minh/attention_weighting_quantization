"""Plot how much assistant answer-token attention goes to text and image tokens."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import get_data_paths, load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.models import load_model  # noqa: E402
from investigation_logging import configure_logging, log  # noqa: E402


# ============================================================================
# CLI AND CONFIGURATION
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse shared config, sampling, output, and logging options."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure, by decoder layer, assistant answer-token attention to image tokens "
            "versus non-image text tokens."
        )
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
    parser.add_argument("--seed", type=int, default=None, help="Override configured sample seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path; defaults to the processed data directory.",
    )
    parser.add_argument("--show", action="store_true", help="Display the graph after saving it.")
    parser.add_argument(
        "--log-level",
        choices=("none", "normal", "extensive"),
        default="normal",
        help="Output detail: none, concise progress, or detailed diagnostics and timings.",
    )
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress Python and dependency warnings.")
    parser.add_argument("--timing", action="store_true", help="Print elapsed time for each phase and sample.")
    return parser.parse_args()


# ============================================================================
# PHASE 3. REGISTER HOOKS
# ============================================================================
def register_attention_hooks(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture per-layer attention mass to image and text keys.

    Args:
        model: Loaded LLaVA model with decoder self-attention layers.
        active_sample: State schema with boolean CPU masks
            ``assistant_query_mask``, ``image_mask``, and ``text_key_mask``,
            each [sequence], plus ``layer_mass``. For each layer, layer_mass
            stores scalar image/text mass sums and a head-query row count for
            the current sample; reset it before every forward.

    Returns:
        Hook handles. The caller removes them in a ``finally`` block.

    Hook contract:
        Read post-softmax attention [batch, heads, query, key], select assistant
        answer queries and image/text keys, then store scalar mass sums. Return
        the original attention output with its full matrix replaced by None so
        the model does not retain the matrices.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find LLaVA decoder layers at model.model.layers")
    # handles: list[torch.utils.hooks.RemovableHandle], registered hooks for caller cleanup.
    handles = []
    for layer_index, layer in enumerate(layers):
        attention_module = getattr(layer, "self_attn", None)
        if attention_module is None:
            raise RuntimeError(f"Decoder layer {layer_index} has no self_attn module")

        def save_attention(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            index: int = layer_index,
        ) -> Any:
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            # weights: torch.Tensor, [batch, heads, query_sequence, key_sequence], post-softmax attention.
            weights = output[1]
            if not torch.is_tensor(weights) or weights.ndim != 4:
                return output

            # query_mask: torch.Tensor, bool [sequence] on weights.device, assistant answer positions.
            query_mask = active_sample["assistant_query_mask"].to(weights.device)
            # image_mask: torch.Tensor, bool [sequence] on weights.device, image key positions.
            image_mask = active_sample["image_mask"].to(weights.device)
            # text_mask: torch.Tensor, bool [sequence] on weights.device, non-image key positions.
            text_mask = active_sample["text_key_mask"].to(weights.device)
            if weights.shape[-2] != query_mask.numel() or weights.shape[-1] != image_mask.numel():
                raise RuntimeError(
                    f"Layer {index} attention shape {tuple(weights.shape)} does not match token masks"
                )

            # Select batch item 0 and assistant queries: [heads, answer_queries, keys].
            # selected: torch.Tensor, float [heads, answer_queries, keys], attention for batch item 0.
            selected = weights[0][:, query_mask, :].float()
            # image_mass: torch.Tensor, scalar, summed attention to image keys.
            image_mass = selected[:, :, image_mask].sum()
            # text_mass: torch.Tensor, scalar, summed attention to text keys.
            text_mass = selected[:, :, text_mask].sum()
            # Per-layer sample schema: summed scalar attention mass by key type
            # and the number of head/query rows contributing to those sums.
            # active_sample["layer_mass"][index]: dict[str, float | int], per-layer scalar masses and row count.
            active_sample["layer_mass"][index] = {
                "image": float(image_mass.cpu()),
                "text": float(text_mass.cpu()),
                "query_rows": int(selected.shape[0] * selected.shape[1]),
            }
            # The layer hook already reduced the matrix to two scalar masses.
            # Avoid retaining full attention matrices in model outputs.
            return (output[0], None, *output[2:])

        handles.append(attention_module.register_forward_hook(save_attention))

    return handles


# ============================================================================
# PHASE 5. ANALYSE AND SAVE RESULTS
# ============================================================================
def calculate_layer_percentages(
    image_mass_by_layer: dict[int, float],
    text_mass_by_layer: dict[int, float],
) -> tuple[list[int], list[float], list[float]]:
    """Convert accumulated per-layer attention masses to image/text shares.

    Args:
        image_mass_by_layer: Layer index -> summed image-key attention mass.
        text_mass_by_layer: Layer index -> summed text-key attention mass.

    Returns:
        Sorted layer indices and matching [layers] image and text percentages.
        Each pair sums to 100 percent.
    """
    # layer_indices: list[int], sorted decoder-layer keys shared by both mass maps.
    layer_indices = sorted(image_mass_by_layer)
    if layer_indices != sorted(text_mass_by_layer):
        raise ValueError("Image and text attention masses have different layer keys")
    # image_percentages: list[float], image attention share for each layer [layers].
    image_percentages: list[float] = []
    # text_percentages: list[float], text attention share for each layer [layers].
    text_percentages: list[float] = []
    for layer_index in layer_indices:
        # total_mass: float, layer's summed attention mass over image and text keys.
        total_mass = image_mass_by_layer[layer_index] + text_mass_by_layer[layer_index]
        if total_mass <= 0:
            raise RuntimeError(f"Layer {layer_index} captured no attention mass")
        # Each appended value is a scalar share; output vectors are [layers].
        image_percentages.append(100.0 * image_mass_by_layer[layer_index] / total_mass)
        text_percentages.append(100.0 * text_mass_by_layer[layer_index] / total_mass)
    return layer_indices, image_percentages, text_percentages


def save_attention_distribution(
    layer_indices: list[int],
    image_percentages: list[float],
    text_percentages: list[float],
    query_rows_by_layer: dict[int, int],
    output_path: Path,
    *,
    sample_count: int,
    seed: int | None,
    show: bool,
) -> Path:
    """Save the stacked percentage graph and per-layer CSV.

    Args:
        layer_indices: Sorted decoder-layer indices [layers].
        image_percentages: Image-key percentages [layers].
        text_percentages: Text-key percentages [layers].
        query_rows_by_layer: Layer index -> contributing head/query row count.
        output_path: PNG destination; CSV uses the same stem and a ``.csv`` suffix.
        sample_count: Number of samples represented by the graph.
        seed: Sample-selection seed shown in the graph title.
        show: Whether to display the graph after saving it.

    Returns:
        Path to the written CSV file.

    Side effects:
        Creates the output directory, writes a PNG and CSV, and optionally
        displays a Matplotlib figure. The figure is always closed afterward.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # csv_path: pathlib.Path, companion CSV destination using the PNG stem.
    csv_path = output_path.with_suffix(".csv")
    figure, axis = plt.subplots(figsize=(max(11, len(layer_indices) * 0.38), 6))
    # Plot vectors have shape [layers]; stacked segments sum to 100% per layer.
    axis.bar(layer_indices, text_percentages, label="Text tokens", color="#2878B5")
    axis.bar(
        layer_indices,
        image_percentages,
        bottom=text_percentages,
        label="Image tokens",
        color="#E87500",
    )
    axis.set_ylim(0, 100)
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Assistant attention (%)")
    axis.set_title(
        f"Assistant answer-token attention to text and image tokens "
        f"({sample_count} samples; seed={seed})"
    )
    axis.set_xticks(layer_indices)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)

    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        # writer: csv.writer, serializes per-layer percentage rows.
        writer = csv.writer(destination)
        writer.writerow(
            ["layer", "text_attention_percent", "image_attention_percent", "assistant_query_head_rows"]
        )
        for layer_index, text_percent, image_percent in zip(
            layer_indices, text_percentages, image_percentages
        ):
            writer.writerow(
                [layer_index, text_percent, image_percent, query_rows_by_layer[layer_index]]
            )
    return csv_path


def main() -> int:
    """Run dataset preparation, model loading, hooks, forwards, and analysis."""
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    try:
        # Read options and validate settings before loading models.
        dataset_config = read_yaml(args.dataset_config)
        model_config = read_yaml(args.model_config)
        data_paths = get_data_paths(args.dataset_config)
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
        # Inputs from setup:
        # args: argparse.Namespace, parsed CLI options and overrides.
        # dataset_config: dict[str, Any], dataset YAML values.
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
        if seed is not None or sample_count < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(sample_count))
        log(f"Selected {len(dataset):,} COCO samples with local images")
        log(
            f"Dataset config={args.dataset_config}, requested={sample_count}, seed={seed}",
            level="extensive",
        )
        if dataset_started is not None:
            log(f"[timing] Prepare dataset: {time.perf_counter() - dataset_started:.2f} s")

        # =====================================================================
        # 2. LOAD MODEL
        # =====================================================================
        # Inputs from phase 1:
        # dataset: datasets.Dataset, selected COCO rows with local images.
        # model_config: dict[str, Any], model YAML values loaded during setup.
        model_started = time.perf_counter() if args.timing else None
        model_id = model_config["model_id"]
        model_dir = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        has_weights = any(model_dir.glob("pytorch_model*.bin")) or any(
            model_dir.glob("model*.safetensors")
        )
        if not (model_dir / "config.json").is_file() or not has_weights:
            model_dir.mkdir(parents=True, exist_ok=True)
            log(f"Downloading LLaVA checkpoint {model_id} to {model_dir}")
            snapshot_download(repo_id=model_id, local_dir=str(model_dir))

        model, input_adapter = load_model(
            model_dir,
            model_config,
            repository_root=REPOSITORY_ROOT,
        )
        model.eval()
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
        # 3. REGISTER HOOKS
        # =====================================================================
        # Inputs from phases 1–2:
        # model: torch.nn.Module, loaded LLaVA model with decoder attention layers.
        # dataset: datasets.Dataset, rows whose assistant queries are analyzed.
        hooks_started = time.perf_counter() if args.timing else None
        # active_sample: dict[str, Any], hook state reset for each sample.
        # active_sample["assistant_query_mask"]: torch.Tensor | None, CPU bool [sequence], answer query mask.
        # active_sample["image_mask"]: torch.Tensor | None, CPU bool [sequence], image key mask.
        # active_sample["text_key_mask"]: torch.Tensor | None, CPU bool [sequence], text key mask.
        # active_sample["layer_mass"]: dict[int, dict[str, float | int]], layer -> mass scalars and row count.
        active_sample: dict[str, Any] = {
            "assistant_query_mask": None,
            "image_mask": None,
            "text_key_mask": None,
            "layer_mass": {},
        }
        hook_handles = register_attention_hooks(model, active_sample)
        log(f"Registered hooks on {len(hook_handles)} decoder self-attention layers")
        if hooks_started is not None:
            log(f"[timing] Register hooks: {time.perf_counter() - hooks_started:.2f} s")

        # =====================================================================
        # 4. FORWARD CALIBRATION SAMPLES
        # =====================================================================
        # Inputs from phase 3:
        # hook_handles: list[torch.utils.hooks.RemovableHandle], hooks removed in finally.
        # active_sample: dict[str, Any], CPU bool masks [sequence] and current layer-mass map.
        # image_mass_by_layer: dict[int, float], cumulative per-layer image-key attention mass.
        image_mass_by_layer: dict[int, float] = {}
        # text_mass_by_layer: dict[int, float], cumulative per-layer text-key attention mass.
        text_mass_by_layer: dict[int, float] = {}
        # query_rows_by_layer: dict[int, int], cumulative head/query row count per layer.
        query_rows_by_layer: dict[int, int] = {}
        try:
            for sample_index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                # row: dict[str, Any], QIG adapter input with conversations, sample id, and local-image sentinel.
                row = {
                    "conversations": sample["conversations"],
                    "id": str(sample.get("id", sample_index)),
                    "image": "local",
                }
                image = sample["image"]
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                # prepared: Any, adapter-preprocessed single-sample row.
                prepared = input_adapter.preprocess_data([image], row)
                # batch: dict[str, Any], collated multimodal sample for input generation.
                batch = input_adapter.data_collator([prepared])
                # prompt_inputs: dict[str, Any], model inputs including embeddings [batch, sequence, hidden].
                # prompt_kwargs: dict[str, Any], adapter attention/vision masks and labels.
                prompt_inputs, prompt_kwargs = input_adapter.generate_input(batch)

                # Adapter fields can be split across CPU and GPU. Keep the
                # masks together on CPU; the attention hook moves them to the
                # weights' device before indexing.
                # attention_mask: torch.Tensor, CPU bool [sequence], valid token positions.
                attention_mask = prompt_kwargs["attention_mask"][0].bool().detach().cpu()
                # image_mask: torch.Tensor, CPU bool [sequence], valid image-token positions.
                image_mask = (
                    prompt_kwargs["vision_mask"][0].bool().detach().cpu() & attention_mask
                )
                # text_key_mask: torch.Tensor, CPU bool [sequence], valid non-image key positions.
                text_key_mask = attention_mask & ~image_mask
                # labels: torch.Tensor, CPU [sequence], answer token ids with -100 ignored.
                labels = prompt_kwargs["labels"][0].detach().cpu()
                # Labels mark answer positions; mask tensors remain CPU bool [sequence].
                # assistant_query_mask: torch.Tensor, CPU bool [sequence], answer positions used as queries.
                assistant_query_mask = labels.ne(-100) & attention_mask
                if not image_mask.any():
                    raise ValueError(f"Sample {row['id']} has no image tokens")
                if not assistant_query_mask.any():
                    raise ValueError(f"Sample {row['id']} has no assistant answer tokens")
                # active_sample["assistant_query_mask"]: CPU bool [sequence], reset to this sample's answer positions.
                active_sample["assistant_query_mask"] = assistant_query_mask
                # active_sample["image_mask"]: CPU bool [sequence], reset to this sample's image positions.
                active_sample["image_mask"] = image_mask
                # active_sample["text_key_mask"]: CPU bool [sequence], reset to this sample's text key positions.
                active_sample["text_key_mask"] = text_key_mask
                # active_sample["layer_mass"]: dict[int, dict[str, float | int]], reset for the current forward.
                active_sample["layer_mass"] = {}
                log(
                    f"Sample {sample_index}/{len(dataset)} id={row['id']}: "
                    f"embeds={tuple(prompt_inputs['inputs_embeds'].shape)}, "
                    f"assistant_queries={int(assistant_query_mask.sum())}, "
                    f"image_keys={int(image_mask.sum())}, text_keys={int(text_key_mask.sum())}",
                    level="extensive",
                )

                forward_started = time.perf_counter() if args.timing else None
                with torch.inference_mode():
                    # outputs: ModelOutput, logits [batch, sequence, vocabulary] plus forward metadata.
                    outputs = input_adapter(
                        inputs_embeds=prompt_inputs["inputs_embeds"],
                        attention_mask=prompt_kwargs["attention_mask"],
                        labels=prompt_kwargs["labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                if forward_started is not None:
                    log(f"[timing] Sample {sample_index} forward: {time.perf_counter() - forward_started:.2f} s")
                expected_layers = len(model.model.layers)
                # missing: list[int], expected decoder layer indices without hook captures.
                missing = sorted(set(range(expected_layers)) - set(active_sample["layer_mass"]))
                if missing:
                    raise RuntimeError(
                        f"No post-softmax attention weights captured for layers {missing}; use eager attention"
                    )
                # mass: dict[str, float | int], one hook's image/text sums and head/query row count.
                for layer_index, mass in active_sample["layer_mass"].items():
                    # image_mass_by_layer[layer_index]: float, accumulated image attention sum.
                    image_mass_by_layer[layer_index] = image_mass_by_layer.get(layer_index, 0.0) + mass["image"]
                    # text_mass_by_layer[layer_index]: float, accumulated text attention sum.
                    text_mass_by_layer[layer_index] = text_mass_by_layer.get(layer_index, 0.0) + mass["text"]
                    # query_rows_by_layer[layer_index]: int, accumulated assistant head/query rows.
                    query_rows_by_layer[layer_index] = query_rows_by_layer.get(layer_index, 0) + mass["query_rows"]
                # sample_image_mass: float, total captured image attention over all layers for this sample.
                sample_image_mass = sum(item["image"] for item in active_sample["layer_mass"].values())
                # sample_text_mass: float, total captured text attention over all layers for this sample.
                sample_text_mass = sum(item["text"] for item in active_sample["layer_mass"].values())
                # sample_image_percent: float, sample-wide image share on a 0–100 scale.
                sample_image_percent = 100.0 * sample_image_mass / (sample_image_mass + sample_text_mass)
                log(
                    f"Sample {sample_index}/{len(dataset)}: layers={len(active_sample['layer_mass'])}, "
                    f"mean image attention={sample_image_percent:.2f}%, "
                    f"mean text attention={100.0 - sample_image_percent:.2f}%",
                    level="extensive",
                )
                if sample_started is not None:
                    log(f"[timing] Sample {sample_index} total: {time.perf_counter() - sample_started:.2f} s")
                del outputs, prompt_inputs, prompt_kwargs, batch, prepared, sample
        finally:
            for handle in hook_handles:
                handle.remove()

        # =====================================================================
        # 5. ANALYSE CAPTURED DATA
        # =====================================================================
        # Inputs from phase 4:
        # image_mass_by_layer: dict[int, float], summed per-layer mass sent to image keys.
        # text_mass_by_layer: dict[int, float], summed per-layer mass sent to text keys.
        # query_rows_by_layer: dict[int, int], assistant head/query rows summed across samples.
        analysis_started = time.perf_counter() if args.timing else None
        # layer_indices: list[int], sorted decoder layers; percentage lists are matching [layers] vectors.
        layer_indices, image_percentages, text_percentages = calculate_layer_percentages(
            image_mass_by_layer,
            text_mass_by_layer,
        )

        # output_path: pathlib.Path, resolved PNG destination for the graph.
        output_path = args.output.expanduser() if args.output else repository_path(
            dataset_config.get(
                "assistant_attention_output",
                data_paths["processed_dir"] / "assistant_attention_distribution.png",
            )
        )
        if not output_path.is_absolute():
            output_path = (REPOSITORY_ROOT / output_path).resolve()
        # csv_path: pathlib.Path, companion CSV written by the analysis helper.
        csv_path = save_attention_distribution(
            layer_indices,
            image_percentages,
            text_percentages,
            query_rows_by_layer,
            output_path,
            sample_count=len(dataset),
            seed=seed,
            show=args.show,
        )

        log(f"Saved graph to {output_path}")
        log(f"Saved per-layer percentages to {csv_path}")
        log(
            "Attention percentages average attention mass over heads and assistant answer-token "
            "queries; text keys include all valid non-image tokens.",
            level="extensive",
        )
        if analysis_started is not None:
            log(f"[timing] Analyze and save graph: {time.perf_counter() - analysis_started:.2f} s")
        if total_started is not None:
            log(f"[timing] Total: {time.perf_counter() - total_started:.2f} s")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
