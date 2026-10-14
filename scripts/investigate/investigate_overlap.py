"""Compare IGA and image-value gradient rankings on generated COCO answers."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.models import load_model  # noqa: E402
from investigation_logging import configure_logging, log  # noqa: E402


# ============================================================================
# PHASE 3. REGISTER HOOKS
# ============================================================================
def register_overlap_hooks(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
    layer_indices: list[int],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture selected value-projection tensors and reduce attention to IGA.

    Args:
        model: Frozen LLaVA causal language model with decoder layers under
            ``model.model.layers``.
        active_sample: Per-sample mutable state initialized with ``capture``
            (bool), CPU ``image_mask`` and ``assistant_query_mask`` (bool
            vectors), ``iga`` (layer -> CPU float32 [image_tokens]), and
            ``value_tensors`` (layer -> live [1, sequence, hidden] tensors).
        layer_indices: Decoder layers for which to collect both scores.

    Returns:
        Hook handles; the caller must remove each handle in ``finally``.

    Hook contracts:
        The ``v_proj`` hook stores its unchanged tensor output at selected
        layers so gradients can later be taken with respect to each image
        token's concatenated value-head vector. The ``self_attn`` hook reads
        post-softmax attention, averages over heads and generated assistant
        query tokens for each image key (IGA), then returns the original
        attention output with only the large attention matrix replaced by
        ``None``. Both hooks are inactive during greedy generation.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find decoder layers at model.model.layers")
    invalid = [index for index in layer_indices if index < 0 or index >= len(layers)]
    if invalid:
        raise ValueError(
            f"Requested overlap layers {invalid}, but model has {len(layers)} decoder layers"
        )

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index in layer_indices:
        layer = layers[layer_index]
        attention = getattr(layer, "self_attn", None)
        value_projection = getattr(attention, "v_proj", None)
        if attention is None or value_projection is None:
            raise RuntimeError(f"Decoder layer {layer_index} has no self_attn.v_proj module")

        def capture_value(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            index: int = layer_index,
        ) -> None:
            if not active_sample["capture"]:
                return
            if not torch.is_tensor(output) or output.ndim != 3 or not output.requires_grad:
                shape = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
                raise RuntimeError(
                    f"Layer {index} value projection must produce a grad-enabled "
                    f"[batch, sequence, hidden] tensor; got {shape}"
                )
            # active_sample["value_tensors"][index]: Tensor, live [1, sequence, hidden] V projection.
            active_sample["value_tensors"][index] = output

        def capture_attention(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            index: int = layer_index,
        ) -> Any:
            if not active_sample["capture"]:
                return output
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            # weights: Tensor, [batch, heads, query_sequence, key_sequence], post-softmax probabilities.
            weights = output[1]
            if not torch.is_tensor(weights) or weights.ndim != 4:
                return output
            # query_mask: Tensor, device bool [sequence], generated assistant query positions.
            query_mask = active_sample["assistant_query_mask"].to(weights.device)
            # image_mask: Tensor, device bool [sequence], image key positions in the expanded multimodal prompt.
            image_mask = active_sample["image_mask"].to(weights.device)
            if weights.shape[-2] != query_mask.numel() or weights.shape[-1] != image_mask.numel():
                raise RuntimeError(
                    f"Layer {index} attention shape {tuple(weights.shape)} does not match token masks"
                )
            if not query_mask.any() or not image_mask.any():
                raise RuntimeError("IGA needs at least one assistant query and one image key")
            # text_to_image: Tensor, float [heads, generated_queries, image_tokens], selected attention.
            text_to_image = weights[0][:, query_mask, :][:, :, image_mask].detach().float()
            # active_sample["iga"][index]: Tensor, CPU float32 [image_tokens], mean attention by image key.
            active_sample["iga"][index] = text_to_image.mean(dim=(0, 1)).cpu()
            return (output[0], None, *output[2:])

        handles.append(value_projection.register_forward_hook(capture_value))
        handles.append(attention.register_forward_hook(capture_attention))

    return handles


# ============================================================================
# PHASE 4. GENERATE GREEDY ANSWERS, THEN FORWARD AND BACKPROPAGATE
# ============================================================================
def generate_greedy_answer(
    input_adapter: Any,
    image: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int,
) -> tuple[torch.Tensor, str]:
    """Greedily generate one answer with caching and no retained gradients.

    The returned token IDs are CPU long [1, generated_tokens], including EOS
    when the model emits it. The generation cache and forward outputs are
    local to this function and released before the analysis forward begins.
    """
    # prepared_prompt: Any, adapter-preprocessed prompt through the latest user message.
    prepared_prompt = input_adapter.preprocess_data([image], prompt_row)
    # prompt_batch: dict[str, Any], collated prompt input for generate_input.
    prompt_batch = input_adapter.data_collator([prepared_prompt])
    # prompt_inputs: dict[str, Tensor], multimodal prompt embeddings [1, prompt_tokens, hidden].
    # prompt_kwargs: dict[str, Tensor], expanded prompt masks and adapter metadata.
    prompt_inputs, prompt_kwargs = input_adapter.generate_input(prompt_batch)
    # current_embeds: Tensor, model-device [1, prompt_tokens, hidden] initially, then newest token [1, 1, hidden].
    current_embeds = prompt_inputs["inputs_embeds"]
    # attention_mask: Tensor, model-device bool [1, cached_tokens + current_tokens].
    attention_mask = prompt_kwargs["attention_mask"]
    # generated_parts: list[Tensor], device-local long [1, 1] greedy token IDs.
    generated_parts: list[torch.Tensor] = []
    # past_key_values: optional model KV cache retained only during this greedy pass.
    past_key_values = None
    # eos_ids: set[int], model EOS token IDs that stop generation after emission.
    eos_id = getattr(input_adapter.model.config, "eos_token_id", None)
    if eos_id is None:
        eos_id = input_adapter.tokenizer.eos_token_id
    eos_ids = set(eos_id) if isinstance(eos_id, (list, tuple)) else ({eos_id} if eos_id is not None else set())
    embedding_layer = input_adapter.model.get_input_embeddings()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # outputs: ModelOutput, logits and an optional updated KV cache for this prefix.
            outputs = input_adapter.model(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            # next_token_id: Tensor, device-local long [1, 1], greedy argmax continuation.
            next_token_id = outputs.logits[0, -1].argmax(dim=-1).reshape(1, 1)
            generated_parts.append(next_token_id)
            if int(next_token_id[0, 0]) in eos_ids:
                break
            # next_embed: Tensor, model-device [1, 1, hidden], embedding for next cached forward.
            next_embed = embedding_layer(next_token_id)
            # appended_attention: Tensor, bool [1, 1], valid newly generated token.
            appended_attention = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            # attention_mask: Tensor, bool [1, all cached and current tokens], extended by one valid token.
            attention_mask = torch.cat((attention_mask, appended_attention), dim=1)
            past_key_values = outputs.past_key_values
            if past_key_values is None:
                # current_embeds: Tensor, [1, full_prefix_tokens, hidden], uncached generation fallback.
                current_embeds = torch.cat((current_embeds, next_embed), dim=1)
            else:
                # current_embeds: Tensor, [1, 1, hidden], only newest input when KV cache is active.
                current_embeds = next_embed

    # generated_ids: Tensor, CPU long [1, generated_tokens], exact emitted IDs including EOS if present.
    generated_ids = torch.cat(generated_parts, dim=1).detach().cpu()
    # assistant_text: str, decoded answer with special tokens omitted for readable progress logging.
    assistant_text = input_adapter.tokenizer.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    del outputs, generated_parts, past_key_values, current_embeds, prompt_inputs, prompt_kwargs, prompt_batch, prepared_prompt
    return generated_ids, assistant_text


def score_generated_answer(
    input_adapter: Any,
    image: Any,
    prompt_row: dict[str, Any],
    generated_ids: torch.Tensor,
    active_sample: dict[str, Any],
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Compute mean generated-answer CE and return per-layer IGA and V-gradient scores."""
    # prepared_prompt: Any, adapter-preprocessed prompt-only multimodal row.
    prepared_prompt = input_adapter.preprocess_data([image], prompt_row)
    # prompt_batch: dict[str, Any], collated prompt-only input.
    prompt_batch = input_adapter.data_collator([prepared_prompt])
    # prompt_inputs/prompt_kwargs: adapter multimodal embeddings [1, prompt_tokens, hidden] and masks.
    prompt_inputs, prompt_kwargs = input_adapter.generate_input(prompt_batch)
    # prompt_embeds: Tensor, device [1, prompt_tokens, hidden], image-expanded prompt representations.
    prompt_embeds = prompt_inputs["inputs_embeds"]
    # prompt_length: int, decoder positions occupied by the multimodal prompt.
    prompt_length = int(prompt_embeds.shape[1])
    # embedding_layer: Module, model token embedding table used to append greedy answer tokens.
    embedding_layer = input_adapter.model.get_input_embeddings()
    # answer_ids: Tensor, embedding-device long [1, answer_tokens], exact greedy IDs generated in phase 4.
    answer_ids = generated_ids.to(device=embedding_layer.weight.device, dtype=torch.long)
    # answer_embeds: Tensor, prompt-device [1, answer_tokens, hidden], value-independent token embeddings.
    answer_embeds = embedding_layer(answer_ids).to(device=prompt_embeds.device)
    # forward_embeds: Tensor, grad-enabled [1, prompt_tokens + answer_tokens, hidden] leaf.
    forward_embeds = torch.cat((prompt_embeds, answer_embeds), dim=1).detach().requires_grad_(True)
    # prompt_attention: Tensor, device bool [1, prompt_tokens], valid multimodal prompt positions.
    prompt_attention = prompt_kwargs["attention_mask"].to(device=forward_embeds.device).bool()
    # answer_attention: Tensor, bool [1, answer_tokens], marks each appended answer token valid.
    answer_attention = torch.ones(
        (1, answer_ids.shape[1]), dtype=prompt_attention.dtype, device=prompt_attention.device
    )
    # forward_attention: Tensor, bool [1, prompt_tokens + answer_tokens], full teacher-forced attention mask.
    forward_attention = torch.cat((prompt_attention, answer_attention), dim=1)
    # image_mask: Tensor, CPU bool [prompt_tokens + answer_tokens], prompt image positions plus false answer positions.
    prompt_image_mask = prompt_kwargs["vision_mask"][0].detach().bool().cpu()
    image_mask = torch.cat((prompt_image_mask, torch.zeros(answer_ids.shape[1], dtype=torch.bool)))
    # assistant_query_mask: Tensor, CPU bool [prompt_tokens + answer_tokens], generated assistant positions as queries.
    assistant_query_mask = torch.zeros(prompt_length + answer_ids.shape[1], dtype=torch.bool)
    assistant_query_mask[prompt_length:] = True
    # active_sample state: reset mask vectors and per-forward score/capture maps before the gradient forward.
    active_sample["image_mask"] = image_mask
    active_sample["assistant_query_mask"] = assistant_query_mask
    active_sample["iga"] = {}
    active_sample["value_tensors"] = {}
    active_sample["capture"] = True
    # ignored_labels: Tensor, model-device long [1, prompt_tokens + answer_tokens], CE supplied manually below.
    ignored_labels = torch.full(
        (1, forward_embeds.shape[1]), -100, dtype=torch.long, device=forward_embeds.device
    )
    model = input_adapter.model
    # output_attentions: bool model setting, enabled only for this scoring pass to control generation memory.
    original_output_attentions = getattr(model.config, "output_attentions", False)
    model.config.output_attentions = True
    try:
        with torch.enable_grad():
            # outputs: ModelOutput, teacher-forced logits [1, prompt + answer, vocabulary].
            outputs = input_adapter(
                inputs_embeds=forward_embeds,
                attention_mask=forward_attention,
                labels=ignored_labels,
                use_cache=False,
                return_dict=True,
            )
            missing_iga = sorted(set(layer_indices) - set(active_sample["iga"]))
            missing_values = sorted(set(layer_indices) - set(active_sample["value_tensors"]))
            if missing_iga:
                raise RuntimeError(
                    f"No attention weights captured at layers {missing_iga}; configure eager attention"
                )
            if missing_values:
                raise RuntimeError(f"No value projections captured at layers {missing_values}")
            # answer_logits: Tensor, float32 [answer_tokens, vocabulary], causal predictions aligned to answer IDs.
            answer_logits = outputs.logits[0, prompt_length - 1 : prompt_length + answer_ids.shape[1] - 1].float()
            if answer_logits.shape[0] != answer_ids.shape[1]:
                raise RuntimeError("Generated answer IDs do not align with causal prediction positions")
            # answer_loss: Tensor, scalar mean CE over generated assistant tokens, including EOS when emitted.
            answer_loss = F.cross_entropy(answer_logits, answer_ids[0].to(answer_logits.device))
            # selected_values: tuple[Tensor], live value-projection outputs for requested layers.
            selected_values = tuple(active_sample["value_tensors"][index] for index in layer_indices)
            # value_gradients: tuple[Tensor], gradient d(mean CE)/d(V) for each requested decoder layer.
            value_gradients = torch.autograd.grad(answer_loss, selected_values)
            # gradient_scores: dict[int, Tensor], CPU float32 [image_tokens], root sum of hidden-dimension squares.
            gradient_scores: dict[int, torch.Tensor] = {}
            for layer_index, value_gradient in zip(layer_indices, value_gradients):
                # image_gradient: Tensor, float32 [image_tokens, hidden], dCE/dV at image positions only.
                image_gradient = value_gradient[0].float()[image_mask.to(value_gradient.device)]
                # gradient_scores[layer]: Tensor, CPU float32 [image_tokens], sqrt(sum_d gradient squared).
                gradient_scores[layer_index] = image_gradient.square().sum(dim=-1).sqrt().detach().cpu()
            iga_scores = {
                layer_index: active_sample["iga"][layer_index]
                for layer_index in layer_indices
            }
    finally:
        model.config.output_attentions = original_output_attentions
        active_sample["capture"] = False
        active_sample["value_tensors"] = {}
    del outputs, answer_logits, answer_loss, selected_values, value_gradients, forward_embeds
    del prompt_embeds, answer_embeds, prompt_inputs, prompt_kwargs, prompt_batch, prepared_prompt
    return iga_scores, gradient_scores


# ============================================================================
# PHASE 5. ANALYSE CAPTURED OVERLAP
# ============================================================================
def calculate_overlap(
    iga_scores: dict[int, torch.Tensor],
    gradient_scores: dict[int, torch.Tensor],
    top_fraction: float,
) -> list[dict[str, int | float]]:
    """Count image-token intersection between per-layer IGA and gradient top sets."""
    rows: list[dict[str, int | float]] = []
    for layer_index in sorted(iga_scores):
        # iga: CPU float32 [image_tokens], mean assistant-query attention per image key.
        iga = iga_scores[layer_index]
        # gradient: CPU float32 [image_tokens], root-sum-square CE gradient per image key.
        gradient = gradient_scores[layer_index]
        if iga.numel() != gradient.numel():
            raise RuntimeError(f"Layer {layer_index} IGA and gradient vectors have different lengths")
        token_count = int(iga.numel())
        if token_count == 0:
            raise RuntimeError(f"Layer {layer_index} has no image tokens")
        # top_count: int, number of image tokens retained independently by both top-fraction rankings.
        top_count = max(1, math.ceil(top_fraction * token_count))
        # iga_top/gradient_top: set[int], image-token indices selected by each metric's top-count values.
        iga_top = set(torch.topk(iga, k=top_count).indices.tolist())
        gradient_top = set(torch.topk(gradient, k=top_count).indices.tolist())
        # overlap_count: int, cardinality of the two top-token set intersections.
        overlap_count = len(iga_top & gradient_top)
        rows.append(
            {
                "layer": layer_index,
                "image_tokens": token_count,
                "top_tokens_per_metric": top_count,
                "overlap_count": overlap_count,
                "overlap_percent_of_top_set": 100.0 * overlap_count / top_count,
            }
        )
    return rows


def save_overlap_results(
    sample_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    show: bool,
) -> Path:
    """Save mean overlap-count graph and companion per-sample/layer CSV."""
    if not sample_rows:
        raise ValueError("No overlap rows were collected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    fields = [
        "sample_id",
        "layer",
        "image_tokens",
        "top_tokens_per_metric",
        "overlap_count",
        "overlap_percent_of_top_set",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    # layer_means: dict[int, float], mean number of overlapping image tokens across samples.
    layer_means = {
        layer: sum(float(row["overlap_count"]) for row in sample_rows if row["layer"] == layer)
        / sum(1 for row in sample_rows if row["layer"] == layer)
        for layer in sorted({int(row["layer"]) for row in sample_rows})
    }
    layers = list(layer_means)
    means = [layer_means[layer] for layer in layers]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(layers, means, color="tab:purple")
    axis.set_xticks(layers)
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Mean overlapping image tokens")
    axis.set_title("Top-token overlap: IGA vs. image-value gradient")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return csv_path


# ============================================================================
# CONFIGURATION AND COMMAND-LINE INTERFACE
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse dataset, model, ranking, output, and logging options."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare top image-token IGA scores with gradients of the image "
            "value cache for greedily generated assistant answers."
        )
    )
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--model-config", type=Path, default=REPOSITORY_ROOT / "configs" / "model.yaml")
    parser.add_argument("--samples", type=int, default=128, help="Number of COCO ShareGPT4V records (default: 128).")
    parser.add_argument("--seed", type=int, default=None, help="Optional calibration subset shuffle seed.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Greedy assistant answer length cap.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[22, 23, 24, 25],
        help="Decoder layers to compare (default: 22 23 24 25).",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=10.0,
        help="Percent of image tokens retained by each metric (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed" / "investigate_overlap.png",
        help="Output graph path; a CSV with the same stem is also written.",
    )
    parser.add_argument("--no-display", action="store_true", help="Save graph without opening a window.")
    parser.add_argument(
        "--log-level",
        choices=("none", "normal", "extensive"),
        default="normal",
        help="Output detail level.",
    )
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress Python/dependency warnings.")
    parser.add_argument("--timing", action="store_true", help="Print elapsed time for each phase and sample.")
    return parser.parse_args()


def main() -> int:
    """Run greedy answer generation, value-cache gradients, IGA, and overlap analysis."""
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    if args.quiet_warnings:
        warnings.filterwarnings("ignore")
    try:
        dataset_config = read_yaml(args.dataset_config)
        model_config = read_yaml(args.model_config)
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if args.samples < 1:
            raise ValueError("--samples must be a positive integer")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be a positive integer")
        if not math.isfinite(args.top_percent) or not 0.0 < args.top_percent <= 100.0:
            raise ValueError("--top-percent must be greater than 0 and at most 100")
        if not args.layers or len(set(args.layers)) != len(args.layers):
            raise ValueError("--layers must contain unique decoder layer indices")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("calibration_seed must be an integer or null")
        top_fraction = args.top_percent / 100.0
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = (REPOSITORY_ROOT / output_path).resolve()
        total_started = time.perf_counter() if args.timing else None

        # ====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # ====================================================================
        dataset_started = time.perf_counter() if args.timing else None
        dataset: Dataset = load_sharegpt4v_dataset(
            source="coco",
            config_path=args.dataset_config,
            download_images=False,
            existing_images_only=True,
        )
        if args.samples > len(dataset):
            raise ValueError(
                f"Requested {args.samples} COCO samples, but only {len(dataset)} "
                "records with local images are available"
            )
        if seed is not None or args.samples < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(args.samples))
        log(f"Selected {len(dataset):,} COCO image-caption pairs with local images")
        log(f"Dataset config={args.dataset_config}, requested={args.samples}, seed={seed}", level="extensive")
        if dataset_started is not None:
            log(f"[timing] Prepare dataset: {time.perf_counter() - dataset_started:.2f} s")

        # ====================================================================
        # 2. LOAD MODEL
        # ====================================================================
        model_started = time.perf_counter() if args.timing else None
        model_id = model_config["model_id"]
        model_dir = repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        has_weights = any(model_dir.glob("pytorch_model*.bin")) or any(model_dir.glob("model*.safetensors"))
        if not (model_dir / "config.json").is_file() or not has_weights:
            model_dir.mkdir(parents=True, exist_ok=True)
            log(f"Downloading LLaVA checkpoint {model_id} to {model_dir}")
            snapshot_download(repo_id=model_id, local_dir=str(model_dir))
        model, input_adapter = load_model(model_dir, model_config, repository_root=REPOSITORY_ROOT)
        model.eval()
        model.requires_grad_(False)
        # output_attentions: bool, kept false for generation and enabled only for each IGA scoring forward.
        model.config.output_attentions = False
        model_parameter = next(model.parameters())
        log(
            f"Loaded model_id={model_id}, model_dir={model_dir}, device={model_parameter.device}, "
            f"dtype={model_parameter.dtype}, layers={len(model.model.layers)}",
            level="extensive",
        )
        if model_started is not None:
            log(f"[timing] Load model: {time.perf_counter() - model_started:.2f} s")

        # ====================================================================
        # 3. REGISTER HOOKS
        # ====================================================================
        # active_sample: dict[str, Any], ``capture`` gate, CPU query/image masks, per-layer IGA, live V tensors.
        active_sample: dict[str, Any] = {
            "capture": False,
            "image_mask": torch.empty(0, dtype=torch.bool),
            "assistant_query_mask": torch.empty(0, dtype=torch.bool),
            "iga": {},
            "value_tensors": {},
        }
        hook_handles = register_overlap_hooks(model, active_sample, args.layers)
        log(f"Registered attention and value hooks on layers {args.layers}")

        # ====================================================================
        # 4. GENERATE GREEDY ANSWERS, THEN FORWARD AND BACKPROPAGATE
        # ====================================================================
        # overlap_rows: list[dict[str, Any]], per-sample/layer top-set intersection statistics for phase 5.
        overlap_rows: list[dict[str, Any]] = []
        try:
            for sample_index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                conversations = sample["conversations"]
                user_indices = [
                    index for index, turn in enumerate(conversations)
                    if turn.get("from") in {"human", "user"}
                ]
                if not user_indices:
                    raise ValueError(f"Sample {sample.get('id', sample_index)} has no user question")
                prompt_conversations = conversations[: user_indices[-1] + 1]
                image = sample["image"].convert("RGB") if hasattr(sample["image"], "convert") else sample["image"]
                prompt_row = {
                    "conversations": prompt_conversations,
                    "id": str(sample.get("id", sample_index)),
                    "image": "local",
                }

                # Disable capture and attention-matrix output during the cached, no-gradient generation pass.
                active_sample["capture"] = False
                model.config.output_attentions = False
                generation_started = time.perf_counter() if args.timing else None
                generated_ids, assistant_text = generate_greedy_answer(
                    input_adapter, image, prompt_row, args.max_new_tokens
                )
                log(f"[{sample_index}/{len(dataset)}] GENERATED ASSISTANT: {assistant_text}", level="normal")
                log(
                    f"Sample {sample_index}: generated_tokens={generated_ids.shape[1]}, "
                    f"answer={assistant_text!r}",
                    level="extensive",
                )
                if generation_started is not None:
                    log(f"[timing] Sample {sample_index} greedy generation: {time.perf_counter() - generation_started:.2f} s")

                # Free GPU KV state as generation helper returned CPU token IDs before the gradient pass.
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                scoring_started = time.perf_counter() if args.timing else None
                iga_scores, gradient_scores = score_generated_answer(
                    input_adapter,
                    image,
                    prompt_row,
                    generated_ids,
                    active_sample,
                    args.layers,
                )
                per_layer_rows = calculate_overlap(iga_scores, gradient_scores, top_fraction)
                sample_id = str(sample.get("id", sample_index))
                for row in per_layer_rows:
                    overlap_rows.append({"sample_id": sample_id, **row})
                    log(
                        f"Sample {sample_index} L{row['layer']}: overlap="
                        f"{row['overlap_count']}/{row['top_tokens_per_metric']} "
                        f"({row['overlap_percent_of_top_set']:.1f}%), image_tokens={row['image_tokens']}",
                        level="extensive",
                    )
                log(
                    f"[{sample_index}/{len(dataset)}] mean overlap="
                    f"{sum(int(row['overlap_count']) for row in per_layer_rows) / len(per_layer_rows):.2f} "
                    f"image tokens across layers {args.layers}",
                )
                if scoring_started is not None:
                    log(f"[timing] Sample {sample_index} scoring/backprop: {time.perf_counter() - scoring_started:.2f} s")
                if sample_started is not None:
                    log(f"[timing] Sample {sample_index} total: {time.perf_counter() - sample_started:.2f} s")
                active_sample["iga"] = {}
                active_sample["value_tensors"] = {}
                del generated_ids, iga_scores, gradient_scores, per_layer_rows, image, sample
        finally:
            active_sample["capture"] = False
            for handle in hook_handles:
                handle.remove()

        # ====================================================================
        # 5. ANALYSE CAPTURED OVERLAP
        # ====================================================================
        analysis_started = time.perf_counter() if args.timing else None
        csv_path = save_overlap_results(overlap_rows, output_path, show=not args.no_display)
        for layer_index in args.layers:
            layer_rows = [row for row in overlap_rows if row["layer"] == layer_index]
            mean_count = sum(float(row["overlap_count"]) for row in layer_rows) / len(layer_rows)
            mean_percent = sum(float(row["overlap_percent_of_top_set"]) for row in layer_rows) / len(layer_rows)
            log(
                f"Layer {layer_index}: mean overlap={mean_count:.2f} image tokens, "
                f"mean top-set overlap={mean_percent:.2f}%"
            )
        log(f"Saved graph to {output_path}")
        log(f"Saved per-sample results to {csv_path}")
        log(
            f"Compared top {args.top_percent:g}% of IGA and value-cache gradient scores; "
            "gradient score is sqrt(sum_d (d(mean answer CE)/dV)^2).",
            level="extensive",
        )
        if analysis_started is not None:
            log(f"[timing] Analyse and save graph: {time.perf_counter() - analysis_started:.2f} s")
        if total_started is not None:
            log(f"[timing] Total: {time.perf_counter() - total_started:.2f} s")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
