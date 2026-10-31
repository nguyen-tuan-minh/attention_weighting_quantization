"""Visualize image-token importance from assistant-token CE gradients."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from attention_quantization.data import load_sharegpt4v_dataset  # noqa: E402
from attention_quantization.models import load_model  # noqa: E402
from attention_quantization.interactive_terminal import (  # noqa: E402
    InteractiveSessionExit,
    InteractiveTimeoutError,
    inspect_generated_token_positions,
)
from investigation_logging import configure_logging, log  # noqa: E402


# ============================================================================
# PHASE 3. REGISTER HOOKS
# ============================================================================
def register_decoder_activation_hooks(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
    capture_point: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture selected decoder activations without changing model computation.

    Args:
        model: Loaded LLaVA causal language model with decoder blocks at
            ``model.model.layers``.
        active_sample: Mutable state whose ``layer_activations`` key maps each
            decoder-layer index to a live tensor [batch, sequence, hidden].
            The mapping is cleared before each sample; captured tensors remain
            attached to the current forward graph.
        capture_point: ``block_input`` for pre-normalization block inputs or
            ``after_layer_norm`` for the output of each block's input norm.

    Returns:
        Hook handles. The caller must remove every handle in a ``finally`` block.

    Hook contract:
        For ``block_input``, read the first positional input to each decoder
        block with a forward-pre-hook. For ``after_layer_norm``, read the
        output of that block's ``input_layernorm`` with a forward hook. Store
        the [batch, sequence, hidden] tensor by layer index; neither hook
        changes model values.
    """
    if capture_point not in {"block_input", "after_layer_norm"}:
        raise ValueError(f"Unsupported activation capture point: {capture_point}")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find decoder blocks at model.model.layers")

    # handles: list[torch.utils.hooks.RemovableHandle], hooks removed after all forwards.
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index, layer in enumerate(layers):
        def store_activation(activation: Any, *, index: int) -> None:
            if not torch.is_tensor(activation) or activation.ndim != 3:
                shape = tuple(activation.shape) if torch.is_tensor(activation) else type(activation).__name__
                raise RuntimeError(
                    f"Decoder layer {index} captured an invalid activation; expected "
                    f"[batch, sequence, hidden], got {shape}"
                )
            if not activation.requires_grad:
                raise RuntimeError(
                    f"Decoder layer {index} activation does not require gradients; "
                    "ensure the model forward runs with gradients enabled"
                )
            # active_sample["layer_activations"][index]: torch.Tensor, live [batch, sequence, hidden] selected activation.
            active_sample["layer_activations"][index] = activation

        if capture_point == "block_input":
            def capture_block_input(
                _module: Any,
                inputs: tuple[Any, ...],
                *,
                index: int = layer_index,
            ) -> None:
                if not active_sample.get("capture_enabled", True):
                    return
                if not inputs:
                    raise RuntimeError(f"Decoder layer {index} received no hidden-state input")
                store_activation(inputs[0], index=index)

            handles.append(layer.register_forward_pre_hook(capture_block_input))
        else:
            layer_norm = getattr(layer, "input_layernorm", None)
            if layer_norm is None:
                raise RuntimeError(f"Decoder layer {layer_index} has no input_layernorm module")

            def capture_after_layer_norm(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not active_sample.get("capture_enabled", True):
                    return
                store_activation(output, index=index)

            handles.append(layer_norm.register_forward_hook(capture_after_layer_norm))

    if not handles:
        raise RuntimeError("No decoder blocks were found in the loaded model")
    return handles


# ============================================================================
# PHASE 4. FORWARD SAMPLES AND BACKPROPAGATE ASSISTANT TOKEN LOSSES
# ============================================================================
def _generate_assistant_once(
    input_adapter: Any,
    image: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int,
    *,
    interactive: bool = False,
    active_sample: dict[str, Any] | None = None,
    sample_number: int = 1,
    heatmap_only: bool = False,
    scale: str = "log",
    score_normalization: str = "raw",
    sampling_mode: str = "greedy",
    temperature: float = 0.8,
    top_p: float = 0.9,
    interactive_timeout_seconds: float = 600.0,
) -> tuple[str, dict[str, Any] | None]:
    """Generate one assistant response and run one interactive inspection pass.

    Args:
        input_adapter: Loaded QIG LLaVA adapter, providing preprocessing,
            collation, multimodal prompt embeddings, model, and tokenizer.
        image: PIL image for the current calibration example.
        prompt_row: Adapter row containing conversation turns up to the latest
            user message, an id, and the local-image sentinel.
        max_new_tokens: Maximum number of generated assistant tokens.
        interactive: Generate the complete answer, then inspect positions and alternatives.
        active_sample: Hook state required to inspect candidate gradient scores.
        sample_number: One-based calibration sample number for candidate heatmaps.
        heatmap_only: Use the separate-reference-image heatmap layout.
        scale: Heatmap color normalization, logarithmic or linear.
        score_normalization: Raw scores or per-layer image-token percentages.
        sampling_mode: ``greedy`` or stochastic nucleus ``sample`` generation.
        temperature: Positive softmax temperature for stochastic generation.
        top_p: Cumulative probability threshold for nucleus sampling.

    Returns:
        Decoded response and an optional interactive action requesting a rerun.

    Side effects:
        Runs repeated model forwards under no-grad mode, stopping on EOS or
        the token limit. It does not write files or retain generation activations.
    """
    # prepared_prompt: Any, preprocessed prompt-only multimodal row.
    prepared_prompt = input_adapter.preprocess_data([image], prompt_row)
    # prompt_batch: dict[str, Any], collated prompt consumed by generate_input.
    prompt_batch = input_adapter.data_collator([prepared_prompt])
    # prompt_inputs: dict[str, torch.Tensor], multimodal prompt embeddings [1, sequence, hidden].
    # prompt_kwargs: dict[str, torch.Tensor], expanded attention mask and all-ignored prompt labels.
    prompt_inputs, prompt_kwargs = input_adapter.generate_input(prompt_batch)
    # prompt_image_mask: torch.Tensor, CPU bool [prompt_sequence], image positions in the multimodal prompt.
    prompt_image_mask = prompt_kwargs["vision_mask"][0].bool().detach().cpu()
    # prompt_attention_mask_cpu: torch.Tensor, CPU bool [prompt_sequence], valid positions for text-label alignment.
    prompt_attention_mask_cpu = prompt_kwargs["attention_mask"][0].bool().detach().cpu()
    # prompt_text_mask: torch.Tensor, CPU bool [prompt_sequence], valid textual positions in the expanded prompt.
    prompt_text_mask = prompt_attention_mask_cpu & ~prompt_image_mask
    # prompt_text_labels: list[str], decoded one-token labels aligned with the prompt's text-mask positions.
    prompt_text_labels = decode_non_image_token_labels(input_adapter, prompt_batch)
    if len(prompt_text_labels) != int(prompt_text_mask.sum()):
        raise RuntimeError("Prompt text token ids do not align with the expanded text-token mask")
    # current_embeds: torch.Tensor, prompt [1, sequence, hidden] or one newest token [1, 1, hidden] with KV cache.
    current_embeds = prompt_inputs["inputs_embeds"]
    # current_attention_mask: torch.Tensor, device-local bool [1, cached_sequence + current_sequence].
    current_attention_mask = prompt_kwargs["attention_mask"]
    # generated_token_ids: list[torch.Tensor], device-local long [1, 1] ids emitted so far.
    generated_token_ids: list[torch.Tensor] = []
    # position_distributions: list[torch.Tensor], CPU float32 [vocabulary] probabilities per generated position.
    position_distributions: list[torch.Tensor] = []
    # past_key_values: model cache for the prompt and generated prefix, or None when unsupported.
    past_key_values = None
    # eos_token_ids: set[int], configured end-of-sequence token ids that stop greedy decoding.
    eos_token_id = getattr(input_adapter.model.config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = input_adapter.tokenizer.eos_token_id
    eos_token_ids = set(eos_token_id) if isinstance(eos_token_id, (tuple, list)) else (
        {eos_token_id} if eos_token_id is not None else set()
    )
    # embedding_layer: torch.nn.Module, model token embedding table for extending prompt embeddings.
    embedding_layer = input_adapter.model.get_input_embeddings()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # outputs: ModelOutput, logits for the current sequence and an optional updated KV cache.
            outputs = input_adapter.model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            # next_token_logits: torch.Tensor, float [vocabulary], distribution for the next assistant token.
            next_token_logits = outputs.logits[0, -1].float()
            # next_probabilities: torch.Tensor, device-local float [vocabulary], full unfiltered token distribution.
            next_probabilities = torch.softmax(next_token_logits, dim=-1)
            if interactive:
                # position_distributions[position]: CPU float32 [vocabulary], retained for selectable top-k/p views.
                position_distributions.append(next_probabilities.detach().cpu())
            if sampling_mode == "sample":
                # sampling_logits: torch.Tensor, float [vocabulary], temperature-scaled scores for sampling.
                sampling_logits = next_token_logits / temperature
                # sampling_probabilities: torch.Tensor, float [vocabulary], temperature-scaled probability distribution.
                sampling_probabilities = torch.softmax(sampling_logits, dim=-1)
                # sorted_probabilities/sorted_ids: descending probability distribution and corresponding token ids.
                sorted_probabilities, sorted_ids = torch.sort(sampling_probabilities, descending=True)
                # cumulative_probabilities: torch.Tensor, float [vocabulary], top-p cumulative mass by rank.
                cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=0)
                # remove_after_nucleus: torch.Tensor, bool [vocabulary], tokens ranked after the top-p nucleus.
                remove_after_nucleus = cumulative_probabilities - sorted_probabilities >= top_p
                sampling_probabilities[sorted_ids[remove_after_nucleus]] = 0.0
                sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
                # next_token_id: torch.Tensor, device-local long [1, 1], sampled from the temperature/top-p distribution.
                next_token_id = torch.multinomial(sampling_probabilities, num_samples=1).reshape(1, 1)
            else:
                # next_token_id: torch.Tensor, device-local long [1, 1], greedy highest-probability token.
                next_token_id = next_token_logits.argmax(dim=-1).reshape(1, 1)
            generated_token_ids.append(next_token_id)
            if int(next_token_id[0, 0]) in eos_token_ids:
                break

            # next_token_embed: torch.Tensor, [1, 1, hidden], embedding for the next cached forward.
            next_token_embed = embedding_layer(next_token_id)
            # next_attention: torch.Tensor, bool [1, 1], marks this generated token as valid.
            next_attention = torch.ones(
                (current_attention_mask.shape[0], 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            )
            # current_attention_mask: torch.Tensor, bool [1, total cached and current tokens].
            current_attention_mask = torch.cat((current_attention_mask, next_attention), dim=1)
            past_key_values = outputs.past_key_values
            if past_key_values is None:
                # Without a KV cache, append the new token embedding to the complete prefix for the next forward.
                current_embeds = torch.cat((current_embeds, next_token_embed), dim=1)
            else:
                # With a KV cache, only the newest token embedding is needed for the next forward.
                current_embeds = next_token_embed

    # generated_ids: torch.Tensor, [1, generated_tokens], decoded continuation ids, including EOS if emitted.
    generated_ids = torch.cat(generated_token_ids, dim=1)
    # assistant_text: str, decoded generated continuation with tokenizer special tokens removed.
    assistant_text = input_adapter.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    inspection_action = None
    if interactive:
        if active_sample is None:
            raise RuntimeError("Interactive inspection requires activation-hook state")
        log(f"GENERATED ASSISTANT ({sampling_mode}): {assistant_text}", level="normal")
        # initial_question: str, latest user-turn text currently used to condition this generated answer.
        initial_question = next(
            (
                str(turn.get("value", ""))
                for turn in reversed(prompt_row["conversations"])
                if turn.get("from") in {"human", "user"}
            ),
            "",
        )
        inspection_action = inspect_generated_token_positions(
            input_adapter,
            active_sample,
            prompt_inputs["inputs_embeds"],
            prompt_kwargs["attention_mask"],
            prompt_kwargs["vision_mask"][0].bool().detach().cpu(),
            prompt_text_labels,
            generated_ids,
            position_distributions,
            max_new_tokens,
            initial_question,
            image,
            sample_number,
            heatmap_only=heatmap_only,
            scale=scale,
            score_normalization=score_normalization,
            score_candidate_token_gradients_fn=score_candidate_token_gradients,
            generate_candidate_continuation_fn=generate_candidate_continuation,
            show_gradient_heatmaps_fn=show_gradient_heatmaps,
            show_large_source_image_fn=show_large_source_image,
            log_fn=log,
            interactive_timeout_seconds=interactive_timeout_seconds,
        )
    del generated_ids, generated_token_ids, prompt_inputs, prompt_kwargs, prompt_batch, prepared_prompt
    return assistant_text, inspection_action


def generate_assistant_reply(
    input_adapter: Any,
    image: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int,
    *,
    interactive: bool = False,
    active_sample: dict[str, Any] | None = None,
    sample_number: int = 1,
    heatmap_only: bool = False,
    scale: str = "log",
    score_normalization: str = "raw",
    temperature: float = 0.8,
    top_p: float = 0.9,
    interactive_timeout_seconds: float = 600.0,
) -> str:
    """Generate an answer, handling interactive regenerate and question-edit actions.

    Args:
        input_adapter: Loaded multimodal adapter and model.
        image: PIL input image for the sample.
        prompt_row: Adapter conversation through the initial user question.
        max_new_tokens: Per-generation assistant token cap.
        interactive: Run the two-screen token and candidate interface.
        active_sample: Shared hook state used for candidate gradient inspection.
        sample_number: One-based sample index shown in visualization titles.
        heatmap_only: Display plain patch maps beside a reference image.
        scale: Gradient-plot color scale.
        score_normalization: Raw scores or within-modality percentages.
        temperature: Sampling temperature for interactive stochastic regeneration.
        top_p: Nucleus mass threshold for stochastic generation.

    Returns:
        Final assistant text after the user exits the interactive interface.

    Side effects:
        Regeneration and question-edit actions rerun preprocessing and generation
        from the updated user prompt until the user exits.
    """
    # current_prompt_row: dict[str, Any], independent prompt conversation that may be edited in the UI.
    current_prompt_row = {**prompt_row, "conversations": [dict(turn) for turn in prompt_row["conversations"]]}
    # sampling_mode: str, current decoding mode, initially greedy for predictable first-pass output.
    sampling_mode = "greedy"
    while True:
        # assistant_text/action: str and optional UI request returned by one complete generation pass.
        assistant_text, action = _generate_assistant_once(
            input_adapter,
            image,
            current_prompt_row,
            max_new_tokens,
            interactive=interactive,
            active_sample=active_sample,
            sample_number=sample_number,
            heatmap_only=heatmap_only,
            scale=scale,
            score_normalization=score_normalization,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p,
            interactive_timeout_seconds=interactive_timeout_seconds,
        )
        if action and action["action"] == "quit":
            raise InteractiveSessionExit
        if not action or action["action"] == "answer_replaced":
            prompt_row["conversations"][:] = [
                dict(turn) for turn in current_prompt_row["conversations"]
            ]
            return action.get("assistant_text", assistant_text) if action else assistant_text
        if action["action"] == "regenerate":
            sampling_mode = action["sampling_mode"]
            continue
        if action["action"] == "edit_question":
            # user_turn_indices: list[int], user question turns available in this prompt-only conversation.
            user_turn_indices = [
                index for index, turn in enumerate(current_prompt_row["conversations"])
                if turn.get("from") in {"human", "user"}
            ]
            if not user_turn_indices:
                raise ValueError("Cannot edit a prompt with no user question")
            # selected_user_turn: dict[str, Any], latest user turn receiving the replacement question text.
            selected_user_turn = current_prompt_row["conversations"][user_turn_indices[-1]]
            # image_marker_count: int, image sentinel count preserved from the original multimodal question.
            image_marker_count = str(selected_user_turn.get("value", "")).count("<image>")
            # replacement_question: str, edited question with the original image sentinels restored at its start.
            replacement_question = action["question"].replace("<image>", "").strip()
            if image_marker_count:
                replacement_question = "\n".join(["<image>"] * image_marker_count + [replacement_question])
            selected_user_turn["value"] = replacement_question
            sampling_mode = action["sampling_mode"]
            continue
        raise ValueError(f"Unknown interactive action: {action['action']}")


def decode_non_image_token_labels(input_adapter: Any, batch: dict[str, Any]) -> list[str]:
    """Decode original non-image input ids for labels aligned with expanded text positions.

    Args:
        input_adapter: Loaded QIG adapter exposing its tokenizer.
        batch: Single-sample collator output with input_ids and attention_mask.

    Returns:
        Decoded token pieces for each valid original text id, excluding negative
        image sentinel ids. This order matches valid non-image positions after
        the adapter expands the image sentinel into image embeddings.
    """
    # input_ids: torch.Tensor, CPU long [1, original_sequence], text ids plus negative image sentinels.
    input_ids = batch["input_ids"][0].detach().cpu()
    # input_attention: torch.Tensor, CPU bool [original_sequence], true for original non-padding entries.
    input_attention = batch["attention_mask"][0].bool().detach().cpu()
    # text_ids: list[int], original valid token ids excluding negative image marker ids.
    text_ids = input_ids[input_attention & input_ids.ge(0)].tolist()
    return [
        input_adapter.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in text_ids
    ]


def score_assistant_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_mask: torch.Tensor,
    text_mask: torch.Tensor,
    layer_activations: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[float]]:
    """Sum squared selected-activation gradients for each answer-token CE.

    Args:
        logits: Model output logits [1, sequence, vocabulary] for one sample.
        labels: CPU integer token labels [sequence]; ``-100`` marks positions
            that are not assistant answer targets.
        image_mask: CPU boolean [sequence] mask selecting image tokens.
        text_mask: CPU boolean [sequence] mask selecting valid non-image text tokens.
        layer_activations: Layer index -> live selected activation tensor
            [1, sequence, hidden] captured by phase 3.

    Returns:
        Image and text score dictionaries map each layer to CPU float32 vectors
        over their respective positions. Each score sums squared gradients over
        assistant answer tokens and hidden features. ``token_losses`` contains
        each scalar CE in answer-token order for logging.

    Assumptions:
        The forward batch has one sample, logits and labels have matching
        sequence length, and every captured decoder activation participates in the
        loss graph. This performs one backward calculation per assistant token.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected one-sample logits [1, sequence, vocabulary], got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.numel() != logits.shape[1]:
        raise ValueError("Labels must be [sequence] and align with the model logits")
    if image_mask.ndim != 1 or image_mask.numel() != logits.shape[1]:
        raise ValueError("Image mask must be [sequence] and align with the model logits")
    if text_mask.ndim != 1 or text_mask.numel() != logits.shape[1]:
        raise ValueError("Text mask must be [sequence] and align with the model logits")
    if not layer_activations:
        raise RuntimeError("Activation hooks captured no layer activations")

    # answer_positions: torch.Tensor, CPU long [answer_tokens], shifted label indices to score.
    answer_positions = labels[1:].ne(-100).nonzero(as_tuple=True)[0] + 1
    if answer_positions.numel() == 0:
        raise ValueError("Sample has no assistant answer tokens with valid CE targets")

    layer_indices = sorted(layer_activations)
    captured_activations = [layer_activations[index] for index in layer_indices]
    for layer_index, activation in zip(layer_indices, captured_activations):
        if activation.shape[0] != 1 or activation.shape[1] != logits.shape[1]:
            raise RuntimeError(
                f"Layer {layer_index} activation shape {tuple(activation.shape)} does not align "
                "with the one-sample logits"
            )

    # image_scores: dict[int, torch.Tensor], layer -> CPU float32 zeros [image_tokens], accumulated for this sample.
    image_scores: dict[int, torch.Tensor] = {
        layer_index: torch.zeros(int(image_mask.sum()), dtype=torch.float32)
        for layer_index in layer_indices
    }
    # text_scores: dict[int, torch.Tensor], layer -> CPU float32 zeros [text_tokens], accumulated for this sample.
    text_scores: dict[int, torch.Tensor] = {
        layer_index: torch.zeros(int(text_mask.sum()), dtype=torch.float32)
        for layer_index in layer_indices
    }
    # token_losses: list[float], detached scalar CE values in answer-token order.
    token_losses: list[float] = []
    answer_token_count = int(answer_positions.numel())

    for token_number, target_position_tensor in enumerate(answer_positions):
        target_position = int(target_position_tensor)
        # target: torch.Tensor, device-local long [1], one assistant token id.
        target = labels[target_position].to(device=logits.device, dtype=torch.long).reshape(1)
        # token_logits: torch.Tensor, float [1, vocabulary], preceding-position logits for this target.
        token_logits = logits[0, target_position - 1].float().unsqueeze(0)
        # token_loss: torch.Tensor, scalar, CE for this single assistant answer token.
        token_loss = F.cross_entropy(token_logits, target)
        token_losses.append(float(token_loss.detach().cpu()))

        # gradients: tuple[torch.Tensor, ...], one [1, sequence, hidden] gradient per selected layer activation.
        gradients = torch.autograd.grad(
            token_loss,
            tuple(captured_activations),
            retain_graph=token_number + 1 < answer_token_count,
            allow_unused=False,
        )
        for layer_index, gradient in zip(layer_indices, gradients):
            # image_gradient: torch.Tensor, float [image_tokens, hidden], this CE's image-token input gradients.
            image_gradient = gradient[0, image_mask.to(gradient.device), :].float()
            # image_scores[layer_index]: torch.Tensor, CPU float32 [image_tokens], sum of squared gradients so far.
            image_scores[layer_index] += image_gradient.square().sum(dim=-1).detach().cpu()
            # text_gradient: torch.Tensor, float [text_tokens, hidden], this CE's valid text-token input gradients.
            text_gradient = gradient[0, text_mask.to(gradient.device), :].float()
            # text_scores[layer_index]: torch.Tensor, CPU float32 [text_tokens], sum of squared gradients so far.
            text_scores[layer_index] += text_gradient.square().sum(dim=-1).detach().cpu()

    return image_scores, text_scores, token_losses


def score_candidate_token_gradients(
    model: torch.nn.Module,
    active_sample: dict[str, Any],
    prefix_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    text_mask: torch.Tensor,
    candidate_token_id: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]:
    """Score one candidate token's CE gradient over image and text positions.

    Args:
        model: Loaded LLaVA causal language model.
        active_sample: Hook state populated with live selected activations during
            this temporary gradient-enabled forward.
        prefix_embeds: Full prompt and accepted-token embeddings [1, sequence, hidden].
        attention_mask: Device-local bool [1, sequence] valid prefix positions.
        image_mask: CPU bool [sequence] positions belonging to the input image.
        text_mask: CPU bool [sequence] valid non-image positions in the full prefix.
        candidate_token_id: Device-local long [1, 1] candidate continuation token.

    Returns:
        Image and text score dictionaries map each layer to CPU float32 vectors
        over their token positions, plus the candidate CE scalar.

    Side effects:
        Temporarily enables and populates the existing activation hooks; clears
        captured graph references and disables capture before returning.
    """
    if image_mask.numel() != prefix_embeds.shape[1]:
        raise ValueError("Image mask must align with the full candidate prefix")
    if text_mask.numel() != prefix_embeds.shape[1]:
        raise ValueError("Text mask must align with the full candidate prefix")
    # active_sample["layer_activations"]: dict[int, torch.Tensor], reset for this candidate's gradient forward.
    active_sample["layer_activations"] = {}
    # active_sample["capture_enabled"]: bool, enable hooks for the candidate-conditioned scoring forward.
    active_sample["capture_enabled"] = True
    # grad_prefix: torch.Tensor, gradient-enabled [1, sequence, hidden] full prompt and chosen prefix embeddings.
    grad_prefix = prefix_embeds.detach().requires_grad_(True)
    try:
        with torch.enable_grad():
            # outputs: ModelOutput, logits for the full candidate prefix.
            outputs = model(
                inputs_embeds=grad_prefix,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            # candidate_loss: torch.Tensor, scalar CE for the candidate at the final prefix position.
            candidate_loss = F.cross_entropy(
                outputs.logits[:, -1, :].float(),
                candidate_token_id.to(device=outputs.logits.device, dtype=torch.long).reshape(1),
            )
            # layer_indices: list[int], decoder layers whose selected activations were captured.
            layer_indices = sorted(active_sample["layer_activations"])
            if not layer_indices:
                raise RuntimeError("Candidate gradient hooks captured no decoder activations")
            # captured_activations: list[torch.Tensor], one live [1, sequence, hidden] tensor per decoder layer.
            captured_activations = [active_sample["layer_activations"][index] for index in layer_indices]
            # gradients: tuple[torch.Tensor, ...], one [1, sequence, hidden] candidate CE gradient per layer.
            gradients = torch.autograd.grad(candidate_loss, tuple(captured_activations))
            # candidate_image_scores: dict[int, torch.Tensor], layer -> CPU float32 [image_tokens] squared-gradient sums.
            candidate_image_scores = {}
            # candidate_text_scores: dict[int, torch.Tensor], layer -> CPU float32 [text_tokens] squared-gradient sums.
            candidate_text_scores = {}
            for layer_index, gradient in zip(layer_indices, gradients):
                # image_gradient: torch.Tensor, float [image_tokens, hidden], candidate's image-position gradient.
                image_gradient = gradient[0, image_mask.to(gradient.device), :].float()
                candidate_image_scores[layer_index] = image_gradient.square().sum(dim=-1).detach().cpu()
                # text_gradient: torch.Tensor, float [text_tokens, hidden], candidate's valid-text-position gradient.
                text_gradient = gradient[0, text_mask.to(gradient.device), :].float()
                candidate_text_scores[layer_index] = text_gradient.square().sum(dim=-1).detach().cpu()
            # candidate_loss_value: float, detached scalar CE for the selected candidate.
            candidate_loss_value = float(candidate_loss.detach().cpu())
    finally:
        # active_sample state: clear graph-owning tensors and disable hooks after this candidate inspection.
        active_sample["layer_activations"] = {}
        active_sample["capture_enabled"] = False
    return candidate_image_scores, candidate_text_scores, candidate_loss_value


def generate_candidate_continuation(
    model: torch.nn.Module,
    embedding_layer: torch.nn.Module,
    prefix_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> tuple[list[int], list[torch.Tensor]]:
    """Greedily generate a candidate-conditioned continuation without gradients.

    Args:
        model: Loaded causal language model.
        embedding_layer: Model token embedding table used to extend the prefix.
        prefix_embeds: Prompt, accepted answer prefix, and selected candidate
            embeddings [1, sequence, hidden].
        attention_mask: Device-local bool mask [1, sequence] for that prefix.
        max_new_tokens: Maximum number of continuation tokens to emit.
        eos_token_ids: End-of-sequence ids that stop the branch after emission.

    Returns:
        Greedy continuation token ids and each step's CPU float32
        [vocabulary] probability distribution, including EOS when emitted.

    Side effects:
        Runs no-gradient model forwards, leaving model parameters and hook state
        unchanged.
    """
    # branch_embeds: torch.Tensor, device-local [1, prefix_tokens, hidden], current counterfactual prefix.
    branch_embeds = prefix_embeds
    # branch_attention_mask: torch.Tensor, bool [1, prefix_tokens], valid positions in the branch prefix.
    branch_attention_mask = attention_mask
    # branch_token_ids: list[int], greedily emitted continuation ids, EOS included if generated.
    branch_token_ids: list[int] = []
    # branch_distributions: list[torch.Tensor], CPU float32 [vocabulary] probabilities before each emitted token.
    branch_distributions: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # branch_outputs: ModelOutput, logits for the current branch prefix.
            branch_outputs = model(
                inputs_embeds=branch_embeds,
                attention_mask=branch_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            # next_probabilities: torch.Tensor, device-local float [vocabulary], raw model distribution for this step.
            next_probabilities = torch.softmax(branch_outputs.logits[0, -1].float(), dim=-1)
            # next_token_id: torch.Tensor, device-local long [1, 1], greedy continuation token.
            next_token_id = next_probabilities.argmax(dim=-1).reshape(1, 1)
            # branch_distributions: append CPU float32 [vocabulary] probabilities aligned to the emitted token.
            branch_distributions.append(next_probabilities.detach().cpu())
            # next_token_value: int, emitted token id used for EOS comparison and decoding.
            next_token_value = int(next_token_id[0, 0])
            branch_token_ids.append(next_token_value)
            if next_token_value in eos_token_ids:
                break
            # next_token_embed: torch.Tensor, device-local [1, 1, hidden], continuation input for the next step.
            next_token_embed = embedding_layer(next_token_id)
            branch_embeds = torch.cat((branch_embeds, next_token_embed), dim=1)
            # branch_attention_mask: torch.Tensor, bool [1, prefix_tokens + 1], marks the appended token valid.
            branch_attention_mask = torch.cat(
                (branch_attention_mask, torch.ones_like(branch_attention_mask[:, :1])), dim=1
            )
    return branch_token_ids, branch_distributions




# ============================================================================
# PHASE 5. DISPLAY AND OPTIONALLY SAVE GRADIENT HEATMAPS
# ============================================================================
def show_large_source_image(source_image: Image.Image, sample_number: int) -> None:
    """Display a sample image alone in a large figure for visual inspection.

    Args:
        source_image: PIL image supplied to the multimodal model.
        sample_number: One-based index in the selected calibration subset.

    Side effects:
        Opens a large Matplotlib figure and closes it after display.
    """
    # image: np.ndarray, [height, width, 3] RGB source image for visual inspection.
    image = np.asarray(source_image.convert("RGB"))
    # figure/axis: Matplotlib figure and axis, dedicated large view of this sample image.
    figure, axis = plt.subplots(figsize=(12, 12), constrained_layout=True)
    axis.imshow(image)
    axis.set_title(f"Sample {sample_number}: input image", fontsize=16)
    axis.axis("off")
    plt.show()
    plt.close(figure)


def show_gradient_heatmaps(
    source_image: Image.Image,
    layer_scores: dict[int, torch.Tensor],
    sample_number: int,
    *,
    text_scores: dict[int, torch.Tensor] | None = None,
    text_labels: list[str] | None = None,
    modality: str = "both",
    heatmap_only: bool = False,
    scale: str = "log",
    score_normalization: str = "raw",
    title_suffix: str | None = None,
    display: bool = True,
    save_dir: Path | None = None,
) -> Path | None:
    """Display image patch maps and aligned text-token score strips by layer.

    Args:
        source_image: Sample image used by the QIG/LLaVA adapter.
        layer_scores: Layer index -> CPU float32 [image_tokens] gradient scores.
        text_scores: Optional layer -> CPU float32 [text_tokens] gradient scores.
        text_labels: Optional decoded text-token pieces aligned with text_scores.
        modality: Draw the image map, text-token strips, or both modalities.
        sample_number: One-based index in the selected calibration subset.
        heatmap_only: Show the source image in a separate reference panel and
            patch-grid scores without image overlays when true.
        scale: Color normalization, either logarithmic or linear.
        score_normalization: ``raw`` preserves score magnitudes; ``percent``
            makes each image-token score a percentage of that layer's total.
        title_suffix: Optional context appended to the figure title.
        display: Whether to display the Matplotlib figure; true by default.
        save_dir: Optional directory enabling PNG output. ``None`` writes no file.

    Returns:
        Saved PNG path, or ``None`` when saving is disabled.

    Side effects:
        Optionally creates ``save_dir`` and writes a PNG, optionally displays a
        Matplotlib figure, and always closes that figure afterward.
    """
    if not layer_scores:
        raise ValueError("No per-layer gradient scores are available to plot")
    if modality not in {"image", "text", "both"}:
        raise ValueError(f"Unknown score modality: {modality}")
    if modality in {"text", "both"} and (not text_scores or text_labels is None):
        raise ValueError("Text plotting requires text scores and aligned token labels")
    if text_scores is not None and text_labels is not None:
        if set(layer_scores) != set(text_scores):
            raise ValueError("Image and text score dictionaries must contain the same layers")
        if any(scores.numel() != len(text_labels) for scores in text_scores.values()):
            raise ValueError("Text labels must align with every layer's text score vector")

    def normalize_scores(scores_by_layer: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if score_normalization == "raw":
            return {layer: scores.float() for layer, scores in scores_by_layer.items()}
        if score_normalization != "percent":
            raise ValueError(f"Unknown score normalization: {score_normalization}")
        normalized = {}
        for layer, scores in scores_by_layer.items():
            total = scores.float().sum()
            normalized[layer] = scores.float() / total * 100.0 if total > 0 else torch.zeros_like(scores, dtype=torch.float32)
        return normalized

    def make_norm(scores_by_layer: dict[int, torch.Tensor]):
        pooled = torch.cat([scores.reshape(-1) for scores in scores_by_layer.values()])
        positives = pooled[pooled > 0]
        if scale == "linear":
            low, high = float(pooled.min()), float(pooled.max())
            return Normalize(vmin=low, vmax=high if high > low else low + 1.0, clip=True)
        if positives.numel() == 0:
            return Normalize(vmin=0.0, vmax=1.0)
        high = float(positives.max())
        low = float(torch.quantile(positives, 0.01))
        return LogNorm(vmin=low if low < high else high * 1e-6, vmax=high, clip=True)

    # show_image/show_text: bool, selects one modality or both for a standalone figure.
    show_image = modality in {"image", "both"}
    show_text = modality in {"text", "both"}
    # image_scores_display/text_scores_display: per-layer CPU float vectors with independent modality color scales.
    image_scores_display = normalize_scores(layer_scores) if show_image else None
    text_scores_display = normalize_scores(text_scores) if show_text and text_scores else None
    if show_image:
        image_token_count = next(iter(image_scores_display.values())).numel()
        grid_size = math.isqrt(image_token_count)
        if grid_size * grid_size != image_token_count:
            raise ValueError(f"Cannot reshape {image_token_count} image-token scores into a square patch grid")
        if any(scores.numel() != image_token_count for scores in image_scores_display.values()):
            raise ValueError("All layers must have the same number of image-token scores")

    source_image = source_image.convert("RGB")
    image_width, image_height = source_image.size
    square_side = max(image_width, image_height)
    background = tuple(round(value * 255) for value in (0.48145466, 0.4578275, 0.40821073))
    square_image = Image.new("RGB", (square_side, square_side), background)
    square_image.paste(source_image, ((square_side - image_width) // 2, (square_side - image_height) // 2))
    image = np.asarray(square_image).astype(np.float32) / 255.0

    layer_indices = sorted(layer_scores if show_image else text_scores)
    image_panel_count = len(layer_indices) + int(heatmap_only) if show_image else 0
    columns = 6
    image_rows = math.ceil(image_panel_count / columns) if show_image else 0
    text_rows = math.ceil(len(layer_indices) / columns) if show_text else 0
    figure = plt.figure(figsize=(columns * 3.5, image_rows * 3.2 + text_rows * 2.5), constrained_layout=True)
    outer = figure.add_gridspec(max(1, image_rows + text_rows), columns)
    image_norm = make_norm(image_scores_display) if show_image else None
    text_norm = make_norm(text_scores_display) if text_scores_display else None
    image_mappable = text_mappable = None
    image_offset = int(heatmap_only) if show_image else 0
    if heatmap_only and show_image:
        ref_axis = figure.add_subplot(outer[0, 0])
        ref_axis.imshow(image)
        ref_axis.set_title("Reference image")
        ref_axis.axis("off")

    image_axes = []
    for layer_number, layer_index in enumerate(layer_indices if show_image else []):
        slot = layer_number + image_offset
        axis = figure.add_subplot(outer[slot // columns, slot % columns])
        image_axes.append(axis)
        grid_scores = image_scores_display[layer_index].reshape(1, 1, grid_size, grid_size)
        if not heatmap_only:
            axis.imshow(image)
            display_heatmap = F.interpolate(grid_scores, size=(square_side, square_side), mode="bilinear", align_corners=False)[0, 0].numpy()
        else:
            display_heatmap = grid_scores[0, 0].numpy()
        image_mappable = axis.imshow(display_heatmap, cmap="inferno", norm=image_norm, alpha=0.58 if not heatmap_only else 1.0)
        axis.set_title(f"Layer {layer_index}")
        axis.axis("off")
    image_slots = image_panel_count
    for slot in range(image_slots, image_rows * columns):
        axis = figure.add_subplot(outer[slot // columns, slot % columns])
        axis.axis("off")
    if image_mappable is not None:
        figure.colorbar(image_mappable, ax=image_axes, fraction=0.015, pad=0.01,
                        label=f"Image-token score ({'percent within image tokens' if score_normalization == 'percent' else 'raw'})")

    if show_text and text_scores_display:
        for layer_number, layer_index in enumerate(layer_indices):
            slot = layer_number
            axis = figure.add_subplot(outer[image_rows + slot // columns, slot % columns])
            values = text_scores_display[layer_index].numpy()
            text_mappable = axis.imshow(values[None, :], aspect="auto", interpolation="nearest", cmap="viridis", norm=text_norm)
            axis.set_title(f"Layer {layer_index}")
            axis.set_yticks([])
            if len(values):
                tick_count = min(12, len(values))
                ticks = np.unique(np.linspace(0, len(values) - 1, tick_count, dtype=int))
                token_tick_labels = [text_labels[index] if text_labels else str(index) for index in ticks]
                axis.set_xticks(ticks)
                axis.set_xticklabels(
                    [label.replace("\n", "\\n")[:10] for label in token_tick_labels],
                    rotation=90,
                    fontsize=6,
                )
        if text_mappable is not None:
            text_axes = figure.axes[-len(layer_indices):]
            figure.colorbar(text_mappable, ax=text_axes, fraction=0.015, pad=0.01,
                            label=f"Text-token score ({'percent within text tokens' if score_normalization == 'percent' else 'raw'})")

    figure_title = f"Sample {sample_number}: selected decoder-activation gradient importance"
    if title_suffix:
        figure_title += f" ({title_suffix})"
    figure.suptitle(figure_title, fontsize=14)

    output_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"sample_{sample_number:04d}_grad_important_{modality}.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        log(f"Saved gradient heatmaps to {output_path}")
    if display:
        plt.show()
    plt.close(figure)
    return output_path


# ============================================================================
# CLI AND CONFIGURATION
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse shared config, sample selection, display/save, and logging options."""
    parser = argparse.ArgumentParser(
        description=(
            "Display image-token importance maps from separate assistant answer-token "
            "cross-entropy gradients at selected decoder activations."
        )
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "dataset.yaml",
    )
    parser.add_argument(
        "--investigate-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "investigate.yaml",
        help="Investigation-specific settings, including interactive terminal timeout.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "model.yaml",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local checkpoint directory overriding model_dir in --model-config.",
    )
    parser.add_argument("--samples", type=int, default=None, help="Override configured sample count.")
    parser.add_argument("--seed", type=int, default=None, help="Override configured sample seed.")
    parser.add_argument(
        "--capture-point",
        choices=("block_input", "after_layer_norm"),
        default="block_input",
        help="Activation to differentiate: pre-norm block input (default) or input-layer-norm output.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("example", "generate", "interactive"),
        default="example",
        help="Score the dataset answer, greedily generate a reply, or inspect generated token positions.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated assistant tokens for generate and interactive modes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature used by interactive stochastic regeneration (default: 0.8).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus probability mass used by interactive stochastic regeneration (default: 0.9).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for per-sample PNGs; omitted means no files are saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open figures; useful when saving with --save-dir in a headless run.",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Show the source image as a reference panel and layer heatmaps without overlays.",
    )
    parser.add_argument(
        "--scale",
        choices=("log", "linear"),
        default="log",
        help="Heatmap color scale: logarithmic (default) or linear.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=("raw", "percent"),
        default="raw",
        help="Show raw scores or each image token's percentage of its layer's total score.",
    )
    parser.add_argument(
        "--log-level",
        choices=("none", "normal", "extensive"),
        default="normal",
        help="Output detail: none, concise progress, or detailed diagnostics and timings.",
    )
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress Python and dependency warnings.")
    parser.add_argument("--timing", action="store_true", help="Print elapsed time for setup and each sample.")
    return parser.parse_args()


def main() -> int:
    """Run dataset setup, model loading, hooks, per-token backprop, and display."""
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    args.timing = args.timing or args.log_level == "extensive"
    try:
        # Read options and validate settings before loading large resources.
        dataset_config = read_yaml(args.dataset_config)
        model_config = read_yaml(args.model_config)
        investigate_config = read_yaml(args.investigate_config)
        interactive_timeout_minutes = investigate_config.get("interactive_timeout_minutes", 10)
        if (
            isinstance(interactive_timeout_minutes, bool)
            or not isinstance(interactive_timeout_minutes, (int, float))
            or not math.isfinite(interactive_timeout_minutes)
            or interactive_timeout_minutes <= 0
        ):
            raise ValueError("interactive_timeout_minutes must be a finite positive number")
        interactive_timeout_seconds = float(interactive_timeout_minutes) * 60.0
        sample_count = (
            args.samples
            if args.samples is not None
            else dataset_config.get("calibration_samples", 2)
        )
        seed = args.seed if args.seed is not None else dataset_config.get("calibration_seed")
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("calibration_samples must be a positive integer")
        if not isinstance(args.max_new_tokens, int) or args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be a positive integer")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError("--temperature must be a finite positive number")
        if not math.isfinite(args.top_p) or not 0.0 < args.top_p <= 1.0:
            raise ValueError("--top-p must be greater than 0 and at most 1")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("calibration_seed must be an integer or null")
        if args.save_dir is not None:
            save_dir = args.save_dir.expanduser()
            if not save_dir.is_absolute():
                save_dir = (REPOSITORY_ROOT / save_dir).resolve()
        else:
            save_dir = None

        total_started = time.perf_counter() if args.timing else None

        # =====================================================================
        # 1. PREPARE CALIBRATION DATASET
        # =====================================================================
        # Inputs from setup:
        # args: argparse.Namespace, parsed sample/config/display options.
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
        model_id = model_config.get("model_id", "")
        model_dir = (
            repository_path(args.model_path)
            if args.model_path is not None
            else repository_path(model_config.get("model_dir", "models/llava-v1.5-7b"))
        )
        has_weights = any(model_dir.glob("pytorch_model*.bin")) or any(
            model_dir.glob("model*.safetensors")
        )
        if not (model_dir / "config.json").is_file() or not has_weights:
            if args.model_path is not None:
                raise FileNotFoundError(
                    f"--model-path must point to a local checkpoint with config.json and model weights: {model_dir}"
                )
            model_dir.mkdir(parents=True, exist_ok=True)
            log(f"Downloading LLaVA checkpoint {model_id} to {model_dir}")
            snapshot_download(repo_id=model_id, local_dir=str(model_dir))

        model, input_adapter = load_model(
            model_dir,
            model_config,
            repository_root=REPOSITORY_ROOT,
        )
        model.eval()
        # Freeze model weights while still allowing gradients through activations that require gradients.
        model.requires_grad_(False)
        model_parameter = next(model.parameters())
        log(
            f"Loaded model_id={model_id or model_dir.name}, model_dir={model_dir}, "
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
        # model: torch.nn.Module, frozen LLaVA model whose selected decoder activations are differentiated.
        # dataset: datasets.Dataset, selected examples processed one at a time.
        hooks_started = time.perf_counter() if args.timing else None
        # active_sample: dict[str, Any], hook state for the current sample and selected forward.
        # active_sample["layer_activations"]: dict[int, torch.Tensor], layer -> live [batch, sequence, hidden] activation.
        # active_sample["capture_enabled"]: bool, false during inference-only answer generation, true for scored forward.
        active_sample: dict[str, Any] = {"layer_activations": {}, "capture_enabled": True}
        hook_handles = register_decoder_activation_hooks(model, active_sample, args.capture_point)
        log(f"Registered {args.capture_point} hooks on {len(hook_handles)} layers")
        if hooks_started is not None:
            log(f"[timing] Register hooks: {time.perf_counter() - hooks_started:.2f} s")

        # =====================================================================
        # 4. FORWARD SAMPLES AND BACKPROPAGATE ASSISTANT TOKEN LOSSES
        # =====================================================================
        # Inputs from phase 3:
        # hook_handles: list[torch.utils.hooks.RemovableHandle], removed in finally after all samples.
        # active_sample["layer_activations"]: dict[int, torch.Tensor], reset per forward and retained for token gradients.
        try:
            for sample_index, sample in enumerate(dataset, start=1):
                sample_started = time.perf_counter() if args.timing else None
                # active_sample["layer_activations"]: dict[int, torch.Tensor], reset before this sample's scored forward.
                active_sample["layer_activations"] = {}
                # source_conversations: list[dict[str, str]], original user/assistant turns from the dataset.
                source_conversations = sample["conversations"]
                # row: dict[str, Any], QIG adapter input with conversation, stable id, and local-image sentinel.
                row = {
                    "conversations": source_conversations,
                    "id": str(sample.get("id", sample_index)),
                    "image": "local",
                }
                image = sample["image"]
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                if args.answer_mode in {"generate", "interactive"}:
                    # user_turn_indices: list[int], positions of user turns available as generation context.
                    user_turn_indices = [
                        index
                        for index, turn in enumerate(source_conversations)
                        if turn.get("from") in {"human", "user"}
                    ]
                    if not user_turn_indices:
                        raise ValueError(f"Sample {row['id']} has no user turn to generate from")
                    # prompt_conversations: list[dict[str, str]], turns through the latest user message, without its answer.
                    prompt_conversations = source_conversations[: user_turn_indices[-1] + 1]
                    # prompt_row: dict[str, Any], user-context row consumed only for assistant generation.
                    prompt_row = {**row, "conversations": prompt_conversations}
                    generation_started = time.perf_counter() if args.timing else None
                    # active_sample["capture_enabled"]: bool, suspend gradient hooks during no-grad generation.
                    active_sample["capture_enabled"] = False
                    try:
                        assistant_text = generate_assistant_reply(
                            input_adapter,
                            image,
                            prompt_row,
                            args.max_new_tokens,
                            interactive=args.answer_mode == "interactive",
                            active_sample=active_sample,
                            sample_number=sample_index,
                            heatmap_only=args.heatmap_only,
                            scale=args.scale,
                            score_normalization=args.score_normalization,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            interactive_timeout_seconds=interactive_timeout_seconds,
                        )
                    finally:
                        # active_sample["capture_enabled"]: bool, re-enable hooks for the answer-scoring forward.
                        active_sample["capture_enabled"] = True
                    if not assistant_text:
                        raise ValueError(f"Sample {row['id']} produced an empty assistant response")
                    if args.answer_mode == "generate":
                        log(
                            f"[{sample_index}/{len(dataset)}] GENERATED ASSISTANT: {assistant_text}",
                            level="normal",
                        )
                    if generation_started is not None:
                        log(
                            f"[timing] Sample {sample_index} generation: "
                            f"{time.perf_counter() - generation_started:.2f} s"
                        )
                    # row["conversations"]: list[dict[str, str]], prompt turns plus generated assistant answer for scoring.
                    row["conversations"] = prompt_conversations + [
                        {"from": "gpt", "value": assistant_text}
                    ]
                # prepared: Any, one adapter-preprocessed multimodal row.
                prepared = input_adapter.preprocess_data([image], row)
                # batch: dict[str, Any], collated single-sample input to generate_input.
                batch = input_adapter.data_collator([prepared])
                # prompt_inputs: dict[str, Any], includes inputs_embeds [1, sequence, hidden].
                # prompt_kwargs: dict[str, Any], adapter attention/vision masks and labels.
                prompt_inputs, prompt_kwargs = input_adapter.generate_input(batch)

                # attention_mask: torch.Tensor, CPU bool [sequence], valid token positions.
                attention_mask = prompt_kwargs["attention_mask"][0].bool().detach().cpu()
                # image_mask: torch.Tensor, CPU bool [sequence], image-token positions restricted to valid tokens.
                image_mask = prompt_kwargs["vision_mask"][0].bool().detach().cpu() & attention_mask
                # labels: torch.Tensor, CPU long [sequence], assistant target ids and -100 ignored positions.
                labels = prompt_kwargs["labels"][0].detach().to(device="cpu", dtype=torch.long)
                answer_token_count = int(labels[1:].ne(-100).sum())
                if not image_mask.any():
                    raise ValueError(f"Sample {row['id']} has no image tokens")
                if answer_token_count == 0:
                    raise ValueError(f"Sample {row['id']} has no assistant answer tokens")

                # inputs_embeds: torch.Tensor, gradient-enabled [1, sequence, hidden] leaf on its adapter device.
                # Detach prevents unnecessary gradient tracking through embedding/vision-tower computation.
                inputs_embeds = prompt_inputs["inputs_embeds"].detach().requires_grad_(True)
                log(
                    f"Sample {sample_index}/{len(dataset)} id={row['id']}: "
                    f"embeds={tuple(inputs_embeds.shape)}, image_tokens={int(image_mask.sum())}, "
                    f"assistant_tokens={answer_token_count}",
                    level="extensive",
                )

                forward_started = time.perf_counter() if args.timing else None
                with torch.enable_grad():
                    # outputs: ModelOutput, logits [1, sequence, vocabulary] and model metadata.
                    outputs = input_adapter(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prompt_kwargs["attention_mask"],
                        # QIG's LLaVA adapter requires labels and uses them only to populate its output loss;
                        # assistant-token gradients below still use the individual logits-derived CEs.
                        labels=prompt_kwargs["labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                    if not active_sample["layer_activations"]:
                        raise RuntimeError("Activation hooks captured no decoder activations")
                    expected_layers = len(model.model.layers)
                    missing_layers = sorted(
                        set(range(expected_layers)) - set(active_sample["layer_activations"])
                    )
                    if missing_layers:
                        raise RuntimeError(f"Activation hooks missed layers {missing_layers}")
                    # logits: torch.Tensor, [1, sequence, vocabulary], causal-LM output used for per-token CE.
                    logits = outputs.logits
                    # forward_layer_activations: dict[int, torch.Tensor], layer -> live selected [1, sequence, hidden] activation.
                    forward_layer_activations = active_sample["layer_activations"]
                    # text_mask: torch.Tensor, CPU bool [sequence], valid non-image tokens aligned to the expanded decoder input.
                    text_mask = prompt_kwargs["attention_mask"][0].bool().detach().cpu() & ~image_mask
                    # text_token_labels: list[str], decoded original text pieces aligned with text_mask positions.
                    text_token_labels = decode_non_image_token_labels(input_adapter, batch)
                    if len(text_token_labels) != int(text_mask.sum()):
                        raise RuntimeError("Input text labels do not align with non-image decoder positions")
                    # layer_scores/text_layer_scores: per-layer CPU float32 image/text position sums of squared gradients.
                    layer_scores, text_layer_scores, token_losses = score_assistant_token_losses(
                        logits,
                        labels,
                        image_mask,
                        text_mask,
                        forward_layer_activations,
                    )
                if forward_started is not None:
                    log(f"[timing] Sample {sample_index} forward and token gradients: {time.perf_counter() - forward_started:.2f} s")
                log(
                    f"[{sample_index}/{len(dataset)}] scored {len(token_losses)} assistant CE tokens; "
                    f"mean CE={sum(token_losses) / len(token_losses):.6g}",
                )

                # Clear references to the completed graph before drawing or forwarding the next sample.
                active_sample["layer_activations"] = {}
                del outputs, logits, forward_layer_activations, inputs_embeds
                plot_started = time.perf_counter() if args.timing else None
                image_output_path = show_gradient_heatmaps(
                    image,
                    layer_scores,
                    sample_index,
                    text_scores=text_layer_scores,
                    text_labels=text_token_labels,
                    modality="image",
                    heatmap_only=args.heatmap_only,
                    scale=args.scale,
                    score_normalization=args.score_normalization,
                    display=not args.no_display,
                    save_dir=save_dir,
                )
                text_output_path = show_gradient_heatmaps(
                    image,
                    layer_scores,
                    sample_index,
                    text_scores=text_layer_scores,
                    text_labels=text_token_labels,
                    modality="text",
                    heatmap_only=args.heatmap_only,
                    scale=args.scale,
                    score_normalization=args.score_normalization,
                    display=not args.no_display,
                    save_dir=save_dir,
                )
                if image_output_path is None and text_output_path is None:
                    log(f"Displayed image and text gradient plots for sample {sample_index}")
                if plot_started is not None:
                    log(f"[timing] Sample {sample_index} plot: {time.perf_counter() - plot_started:.2f} s")
                if sample_started is not None:
                    log(f"[timing] Sample {sample_index} total: {time.perf_counter() - sample_started:.2f} s")
                del layer_scores, text_layer_scores, text_mask, text_token_labels, token_losses, prompt_inputs, prompt_kwargs, batch, prepared, image, sample
        finally:
            for handle in hook_handles:
                handle.remove()

        # =====================================================================
        # 5. DISPLAY AND OPTIONALLY SAVE RESULTS
        # =====================================================================
        # Phase 4 displayed one per-layer heatmap grid per sample; --heatmap-only omits source-image backgrounds.
        # PNG files are written only when ``--save-dir`` was supplied.
        log(f"Gradient-importance analysis complete for {len(dataset):,} samples.")
        if save_dir is not None:
            log(f"Saved per-sample figures under: {save_dir}")
        if total_started is not None:
            log(f"[timing] Total: {time.perf_counter() - total_started:.2f} s")
    except InteractiveTimeoutError:
        print("Interactive session timed out; exiting the investigation.")
        return 0
    except InteractiveSessionExit:
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
