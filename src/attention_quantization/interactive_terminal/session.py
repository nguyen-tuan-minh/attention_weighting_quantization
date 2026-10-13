"""Controller for interactive assistant-token inspection."""

from __future__ import annotations

from typing import Any

import torch

from .all_tokens import read_all_tokens_command, render_all_tokens, show_all_tokens_help
from .candidate_screen import (
    read_candidate_command,
    render_candidates,
    show_candidate_help,
    show_gradient_norm_percentages,
)
from .terminal_io import read_timed_input


def inspect_generated_token_positions(
    input_adapter: Any,
    active_sample: dict[str, Any],
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_image_mask: torch.Tensor,
    prompt_text_labels: list[str],
    generated_ids: torch.Tensor,
    position_distributions: list[torch.Tensor],
    max_new_tokens: int,
    initial_question: str,
    image: Any,
    sample_number: int,
    *,
    heatmap_only: bool,
    scale: str,
    score_normalization: str,
    score_candidate_token_gradients_fn: Any,
    generate_candidate_continuation_fn: Any,
    show_gradient_heatmaps_fn: Any,
    show_large_source_image_fn: Any,
    log_fn: Any,
    interactive_timeout_seconds: float,
) -> dict[str, Any] | None:
    """Inspect stored top-token distributions and candidate gradients by generated position.

    The model generates the full answer before this UI starts. Each stored
    distribution is conditioned on the prefix before its numbered token. The
    user may inspect several positions and optionally display one candidate's
    image and text gradient views for each candidate at a position.
    """
    # token_ids: list[int], generated assistant tokens, including EOS when emitted.
    token_ids = generated_ids[0].tolist()
    # current_assistant_text: str, decoded current answer, refreshed after candidate replacement.
    current_assistant_text = input_adapter.tokenizer.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    while True:
        render_all_tokens(input_adapter, initial_question, token_ids, position_distributions)
        command_parts = read_all_tokens_command(len(token_ids), interactive_timeout_seconds)
        if command_parts is None:
            return {"action": "quit"}
        command_name = command_parts[0].lower() if command_parts else ""
        if command_name == "i" and len(command_parts) == 1:
            show_large_source_image_fn(image, sample_number)
            continue
        if command_name == "q" and len(command_parts) == 1:
            return {"action": "quit"}
        if command_name == "r" and len(command_parts) <= 2:
            sampling_mode = "sample" if len(command_parts) == 2 and command_parts[1].lower() in {"s", "sample"} else "greedy"
            if len(command_parts) == 2 and command_parts[1].lower() not in {"s", "sample", "g", "greedy"}:
                print("Use r for greedy regeneration or r s for stochastic regeneration.")
                continue
            return {"action": "regenerate", "sampling_mode": sampling_mode}
        if command_name == "e" and len(command_parts) == 1:
            try:
                new_question = read_timed_input("New initial question: ", interactive_timeout_seconds).strip()
            except EOFError:
                return {"action": "quit"}
            if not new_question:
                print("Question was empty; keeping the current answer.")
                continue
            while True:
                try:
                    sampling_choice = read_timed_input(
                        "Regenerate with [g]reedy or [s]tochastic sampling? [g]: ",
                        interactive_timeout_seconds,
                    ).strip().lower()
                except EOFError:
                    return {"action": "quit"}
                if sampling_choice in {"", "g", "greedy", "s", "sample", "stochastic"}:
                    break
                print("Choose g for greedy or s for stochastic sampling.")
            return {
                "action": "edit_question",
                "question": new_question,
                "sampling_mode": "sample" if sampling_choice in {"s", "sample", "stochastic"} else "greedy",
            }
        if command_name == "h" and len(command_parts) == 1:
            show_all_tokens_help()
            continue
        if not command_parts or not command_parts[0].isdigit():
            print("Use n, n k count, n p mass, r [s], e, i, h, or q.")
            continue
        # position: int, one-based generated assistant token position selected for inspection.
        position = int(command_parts[0])
        if not 1 <= position <= len(token_ids):
            print(f"Choose a token position from 1 to {len(token_ids)}.")
            continue
        # candidate_distribution: torch.Tensor, CPU float32 [vocabulary], probabilities at this generated position.
        candidate_distribution = position_distributions[position - 1]
        # display_count: int, number of ranked candidates requested, defaulting to ten.
        display_count = 10
        if len(command_parts) == 1:
            pass
        elif len(command_parts) == 3 and command_parts[1].lower() == "k" and command_parts[2].isdigit():
            display_count = int(command_parts[2])
            if not 1 <= display_count <= candidate_distribution.numel():
                print(f"Top-k must be from 1 to {candidate_distribution.numel()}.")
                continue
        elif len(command_parts) == 3 and command_parts[1].lower() == "p":
            # top_mass: float, requested cumulative probability mass, accepted as 0.9 or 90%.
            mass_text = command_parts[2]
            try:
                top_mass = float(mass_text[:-1]) / 100.0 if mass_text.endswith("%") else float(mass_text)
            except ValueError:
                print("Top-p mass must be between 0 and 1, or a percent such as 90%.")
                continue
            if not 0.0 < top_mass <= 1.0:
                print("Top-p mass must be greater than 0 and at most 1.")
                continue
            # sorted_probabilities/sorted_ids: CPU float32 probabilities and ids in descending rank order.
            sorted_probabilities, sorted_ids = torch.sort(candidate_distribution, descending=True)
            # cumulative_mass: CPU float32 [vocabulary], accumulated probability mass by rank.
            cumulative_mass = torch.cumsum(sorted_probabilities, dim=0)
            display_count = min(
                candidate_distribution.numel(),
                int(torch.searchsorted(cumulative_mass, top_mass).item()) + 1,
            )
        else:
            print("Use n, n k count, or n p mass (for example: 27 k 20 or 27 p 0.9).")
            continue
        # top_probabilities/top_token_ids: CPU vectors [display_count], selected ranked candidate distribution.
        if len(command_parts) == 3 and command_parts[1].lower() == "p":
            top_probabilities = sorted_probabilities[:display_count]
            top_token_ids = sorted_ids[:display_count]
        else:
            top_probabilities, top_token_ids = torch.topk(candidate_distribution, k=display_count)
        # position_options: list[tuple[int, float]], ranked candidate ids and their full-distribution probabilities.
        position_options = [
            (int(candidate_id), float(candidate_probability))
            for candidate_id, candidate_probability in zip(top_token_ids, top_probabilities)
        ]
        while True:
            render_candidates(input_adapter, position, position_options)
            command = read_candidate_command(len(position_options), interactive_timeout_seconds)
            if command is None:
                return {"action": "quit"}
            if command == ["q"]:
                return {"action": "quit"}
            if command == ["b"]:
                break
            if command == ["h"]:
                show_candidate_help()
                continue
            if not command or not command[0].isdigit() or len(command) > 3:
                print("Use n, n i, n t, n t c, n g, n + c, n replace [c], b, or q.")
                continue
            # candidate_rank: int, one-based rank in the top candidate list for the selected assistant position.
            candidate_rank = int(command[0])
            if not 1 <= candidate_rank <= len(position_options):
                print(f"Choose a candidate rank from 1 to {len(position_options)}.")
                continue
            if len(command) >= 2 and command[1] == "replace":
                if len(command) == 3 and (not command[2].isdigit() or int(command[2]) < 1):
                    print("Use `n replace c` with a positive continuation token limit.")
                    continue
                if len(command) > 3:
                    print("Use `n replace` or `n replace c`.")
                    continue
                # replacement_suffix_limit: int, optional continuation cap after replacement; default uses remaining total cap.
                replacement_suffix_limit = (
                    int(command[2]) if len(command) == 3 else max(0, max_new_tokens - position)
                )
                # replacement_token_id: torch.Tensor, device-local long [1, 1], selected alternative at this position.
                replacement_token_id = torch.tensor(
                    [[position_options[candidate_rank - 1][0]]],
                    device=prompt_embeds.device,
                    dtype=torch.long,
                )
                # preceding_token_ids: torch.Tensor, device-local long [1, preceding_generated_tokens].
                preceding_token_ids = generated_ids[:, : position - 1]
                # replacement_prefix_embeds: torch.Tensor, [1, prompt_plus_preceding_tokens, hidden] before replacement.
                replacement_prefix_embeds = prompt_embeds
                if preceding_token_ids.shape[1] > 0:
                    # preceding_token_embeds: torch.Tensor, device-local [1, preceding_generated_tokens, hidden].
                    preceding_token_embeds = input_adapter.model.get_input_embeddings()(preceding_token_ids)
                    replacement_prefix_embeds = torch.cat(
                        (prompt_embeds, preceding_token_embeds), dim=1
                    )
                # replacement_prefix_attention: torch.Tensor, bool [1, prompt_plus_preceding_tokens].
                replacement_prefix_attention = torch.cat(
                    (
                        prompt_attention_mask,
                        torch.ones(
                            (1, preceding_token_ids.shape[1]),
                            dtype=prompt_attention_mask.dtype,
                            device=prompt_attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                # replacement_prefix_with_token: torch.Tensor, [1, prefix_tokens + 1, hidden], context including candidate.
                replacement_prefix_with_token = torch.cat(
                    (
                        replacement_prefix_embeds,
                        input_adapter.model.get_input_embeddings()(replacement_token_id),
                    ),
                    dim=1,
                )
                # replacement_attention: torch.Tensor, bool [1, prefix_tokens + 1], valid selected-token context.
                replacement_attention = torch.cat(
                    (
                        replacement_prefix_attention,
                        torch.ones_like(replacement_prefix_attention[:, :1]),
                    ),
                    dim=1,
                )
                # replacement_eos_ids: set[int], EOS markers for stopping regenerated continuation tokens.
                replacement_eos_id = getattr(input_adapter.model.config, "eos_token_id", None)
                if replacement_eos_id is None:
                    replacement_eos_id = input_adapter.tokenizer.eos_token_id
                replacement_eos_ids = set(replacement_eos_id) if isinstance(replacement_eos_id, (tuple, list)) else (
                    {replacement_eos_id} if replacement_eos_id is not None else set()
                )
                if int(replacement_token_id[0, 0]) in replacement_eos_ids:
                    replacement_suffix_limit = 0
                # replacement_suffix_ids/distributions: generated continuation and per-position CPU distributions.
                replacement_suffix_ids, replacement_suffix_distributions = generate_candidate_continuation_fn(
                    input_adapter.model,
                    input_adapter.model.get_input_embeddings(),
                    replacement_prefix_with_token,
                    replacement_attention,
                    replacement_suffix_limit,
                    replacement_eos_ids,
                )
                # replacement_suffix_tensor: device-local long [1, regenerated_suffix_tokens].
                replacement_suffix_tensor = torch.tensor(
                    [replacement_suffix_ids], device=generated_ids.device, dtype=generated_ids.dtype
                )
                # updated_generated_ids: device-local long [1, prefix + replacement + regenerated suffix].
                updated_generated_ids = torch.cat(
                    (
                        preceding_token_ids,
                        replacement_token_id,
                        replacement_suffix_tensor,
                    ),
                    dim=1,
                )
                # replacement_distribution: CPU float32 [vocabulary], original conditional distribution at this position.
                replacement_distribution = position_distributions[position - 1]
                # updated_position_distributions: per-position CPU probability vectors aligned with updated_generated_ids.
                updated_position_distributions = (
                    position_distributions[: position - 1]
                    + [replacement_distribution]
                    + replacement_suffix_distributions
                )
                # generated_ids/token_ids/distributions: replace displayed answer sequence and return screen to all tokens.
                generated_ids = updated_generated_ids
                token_ids = generated_ids[0].tolist()
                position_distributions = updated_position_distributions
                current_assistant_text = input_adapter.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
                print(f"Replaced position [{position}] and regenerated: {current_assistant_text}")
                log_fn(f"GENERATED ASSISTANT (candidate replacement): {current_assistant_text}", level="normal")
                break
            if len(command) >= 2 and command[1] == "+":
                if len(command) != 3 or not command[2].isdigit() or int(command[2]) < 1:
                    print("Use `n + c` with a positive continuation length c.")
                    continue
                # continuation_limit: int, number of next assistant tokens to generate in both branches.
                continuation_limit = int(command[2])
                # prefix_token_ids: torch.Tensor, device-local long [1, preceding_generated_tokens].
                prefix_token_ids = generated_ids[:, : position - 1]
                # branch_prefix_embeds: torch.Tensor, [1, prompt_plus_prior_tokens, hidden] before candidate selection.
                branch_prefix_embeds = prompt_embeds
                if prefix_token_ids.shape[1] > 0:
                    # branch_prior_embeds: torch.Tensor, device-local [1, preceding_generated_tokens, hidden].
                    branch_prior_embeds = input_adapter.model.get_input_embeddings()(prefix_token_ids)
                    branch_prefix_embeds = torch.cat((prompt_embeds, branch_prior_embeds), dim=1)
                # branch_attention_mask: torch.Tensor, bool [1, prompt_plus_prior_tokens], attention mask before candidate.
                branch_attention_mask = torch.cat(
                    (
                        prompt_attention_mask,
                        torch.ones(
                            (1, prefix_token_ids.shape[1]),
                            dtype=prompt_attention_mask.dtype,
                            device=prompt_attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                # selected_candidate_id: torch.Tensor, device-local long [1, 1], chosen counterfactual token id.
                selected_candidate_id = torch.tensor(
                    [[position_options[candidate_rank - 1][0]]],
                    device=prompt_embeds.device,
                    dtype=torch.long,
                )
                # original_token_id: torch.Tensor, device-local long [1, 1], model-generated token at this position.
                original_token_id = generated_ids[:, position - 1 : position]
                # selected_prefix_embeds: torch.Tensor, [1, prefix_tokens + 1, hidden], prefix extended with candidate.
                selected_prefix_embeds = torch.cat(
                    (branch_prefix_embeds, input_adapter.model.get_input_embeddings()(selected_candidate_id)),
                    dim=1,
                )
                # selected_attention_mask: torch.Tensor, bool [1, prefix_tokens + 1], valid counterfactual context.
                selected_attention_mask = torch.cat(
                    (branch_attention_mask, torch.ones_like(branch_attention_mask[:, :1])), dim=1
                )
                # model_eos_ids: set[int], EOS markers for dynamic branch termination.
                model_eos_id = getattr(input_adapter.model.config, "eos_token_id", None)
                if model_eos_id is None:
                    model_eos_id = input_adapter.tokenizer.eos_token_id
                model_eos_ids = set(model_eos_id) if isinstance(model_eos_id, (tuple, list)) else (
                    {model_eos_id} if model_eos_id is not None else set()
                )
                # selected_suffix_ids: list[int], greedy counterfactual tokens following the selected candidate.
                selected_suffix_ids, _ = generate_candidate_continuation_fn(
                    input_adapter.model,
                    input_adapter.model.get_input_embeddings(),
                    selected_prefix_embeds,
                    selected_attention_mask,
                    continuation_limit,
                    model_eos_ids,
                )
                # original_continuation: list[int], up to c already-generated tokens after the original token.
                original_continuation = generated_ids[0, position : position + continuation_limit].tolist()
                if model_eos_ids:
                    for suffix_index, suffix_token_id in enumerate(original_continuation):
                        if int(suffix_token_id) in model_eos_ids:
                            original_continuation = original_continuation[: suffix_index + 1]
                            break
                # selected_candidate_text: str, chosen alternative token piece with special-token spelling retained.
                selected_candidate_text = input_adapter.tokenizer.decode(
                    [int(selected_candidate_id[0, 0])], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                # original_token_text: str, model-generated token piece at the inspected position.
                original_token_text = input_adapter.tokenizer.decode(
                    [int(original_token_id[0, 0])], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                # selected_suffix_text/original_suffix_text: str, printable continuations with special tokens removed.
                selected_suffix_text = input_adapter.tokenizer.decode(
                    selected_suffix_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                original_suffix_text = input_adapter.tokenizer.decode(
                    original_continuation, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                print(f"Original [{original_token_text!r}] then next tokens: {original_suffix_text!r}")
                print(f"Candidate [{selected_candidate_text!r}] then next tokens: {selected_suffix_text!r}")
                continue
            # modality: str, image by default or explicitly selected text-token scores.
            modality = command[1] if len(command) >= 2 else "i"
            if modality not in {"i", "t", "g"} or (modality in {"i", "g"} and len(command) > 2):
                print("Use n, n i, n t, n t c, or n g; c is only valid with text.")
                continue
            # text_limit: int | None, number of most recent prior text tokens to display; None means all.
            text_limit = None
            if len(command) == 3:
                if modality != "t" or not command[2].isdigit() or int(command[2]) < 1:
                    print("The text-token count c must be a positive integer in `n t c`.")
                    continue
                text_limit = int(command[2])
            # prefix_token_ids: torch.Tensor, device-local long [1, preceding_generated_tokens].
            prefix_token_ids = generated_ids[:, : position - 1]
            # prefix_embeds: torch.Tensor, device-local [1, prompt_plus_preceding_tokens, hidden].
            prefix_embeds = prompt_embeds
            if prefix_token_ids.shape[1] > 0:
                # prior_token_embeds: torch.Tensor, device-local [1, preceding_generated_tokens, hidden].
                prior_token_embeds = input_adapter.model.get_input_embeddings()(prefix_token_ids)
                prefix_embeds = torch.cat((prompt_embeds, prior_token_embeds), dim=1)
            # prefix_attention_mask: torch.Tensor, bool [1, prompt_plus_preceding_tokens].
            prefix_attention_mask = torch.cat(
                (
                    prompt_attention_mask,
                    torch.ones(
                        (1, prefix_token_ids.shape[1]),
                        dtype=prompt_attention_mask.dtype,
                        device=prompt_attention_mask.device,
                    ),
                ),
                dim=1,
            )
            # prefix_image_mask: torch.Tensor, CPU bool [prompt_plus_preceding_tokens], image positions only in prompt.
            prefix_image_mask = torch.cat(
                (prompt_image_mask, torch.zeros(prefix_token_ids.shape[1], dtype=torch.bool))
            )
            # prefix_text_mask: torch.Tensor, CPU bool [prompt_plus_preceding_tokens], all valid non-image positions.
            prefix_text_mask = prefix_attention_mask[0].detach().bool().cpu() & ~prefix_image_mask
            # prefix_text_labels: list[str], text-token pieces aligned to prefix_text_mask positions.
            prefix_text_labels = prompt_text_labels + [
                input_adapter.tokenizer.decode(
                    [int(token_id)], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for token_id in prefix_token_ids[0].detach().cpu().tolist()
            ]
            if len(prefix_text_labels) != int(prefix_text_mask.sum()):
                raise RuntimeError("Generated prefix text labels do not align with non-image positions")
            # candidate_id: torch.Tensor, device-local long [1, 1], selected alternative token id.
            candidate_id = torch.tensor(
                [[position_options[candidate_rank - 1][0]]],
                device=prompt_embeds.device,
                dtype=torch.long,
            )
            candidate_image_scores, candidate_text_scores, candidate_loss = score_candidate_token_gradients_fn(
                input_adapter.model,
                active_sample,
                prefix_embeds,
                prefix_attention_mask,
                prefix_image_mask,
                prefix_text_mask,
                candidate_id,
            )
            candidate_text = input_adapter.tokenizer.decode(
                [int(candidate_id[0, 0])],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            layer_totals = ", ".join(
                f"L{layer_index}={float(scores.sum()):.3g}"
                for layer_index, scores in sorted(candidate_image_scores.items())
            )
            text_totals = ", ".join(
                f"L{layer_index}={float(scores.sum()):.3g}"
                for layer_index, scores in sorted(candidate_text_scores.items())
            )
            if text_limit is not None:
                # candidate_text_scores: dict[int, torch.Tensor], retain only the latest c prior text positions for display.
                candidate_text_scores = {
                    layer_index: scores[-text_limit:]
                    for layer_index, scores in candidate_text_scores.items()
                }
                # prefix_text_labels: list[str], token labels corresponding to the retained recent text positions.
                prefix_text_labels = prefix_text_labels[-text_limit:]
            print(
                f"Position [{position}] candidate {candidate_rank} {candidate_text!r}: "
                f"CE={candidate_loss:.5g}; per-layer image sums: {layer_totals}; text sums: {text_totals}"
            )
            if modality == "g":
                show_gradient_norm_percentages(
                    candidate_image_scores,
                    candidate_text_scores,
                    title=(
                        f"Position {position}, candidate {candidate_text!r}: "
                        "image/text gradient norm share by layer"
                    ),
                )
                continue
            show_gradient_heatmaps_fn(
                image,
                candidate_image_scores,
                sample_number,
                text_scores=candidate_text_scores if modality == "t" else None,
                text_labels=prefix_text_labels if modality == "t" else None,
                heatmap_only=heatmap_only,
                scale=scale,
                score_normalization=score_normalization,
                title_suffix=f"position {position}, candidate {candidate_text!r}",
                modality="image" if modality == "i" else "text",
                display=True,
                save_dir=None,
            )
            # Stay with this assistant position after drawing the selected candidate and modality.
