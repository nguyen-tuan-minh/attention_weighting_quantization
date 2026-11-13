"""Interactively inspect POPE questions with LLaVA and optional contrastive decoding."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention_quantization.config import read_yaml, repository_path  # noqa: E402
from contrastive_decoding import (  # noqa: E402
    decode_one,
    eos_token_ids,
    load_model,
    load_runtime,
    parse_memory_size,
    validate_compatibility,
)
from investigation_logging import configure_logging, log  # noqa: E402
from attention_quantization.interactive_terminal.terminal_io import (  # noqa: E402
    InteractiveSessionExit,
    InteractiveTimeoutError,
    read_timed_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--pretrained", "--expert-model", dest="expert_model", default=None,
                        help="Expert checkpoint path; defaults to model.yaml.")
    parser.add_argument("--amateur-model", default=None,
                        help="Optional amateur checkpoint for amateur or contrastive mode.")
    parser.add_argument("--question-id", default=None,
                        help="Initial POPE question_id. Duplicate IDs start at their first dataset row; use --row-id to disambiguate.")
    parser.add_argument("--row-id", default=None,
                        help="Initial unique POPE row id; mutually exclusive with --question-id.")
    parser.add_argument("--dataset", default="lmms-lab/POPE")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrastive-weight", type=float, default=1.0,
                        help="Initial contrastive weight and default for the interactive model command.")
    parser.add_argument("--contrastive-score-type", choices=("logits", "log-probabilities"), default="logits")
    parser.add_argument("--plausibility-threshold", type=float, default=0.001)
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Generation limit; defaults to the LMMS-Eval POPE setting (128).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for `s all` or the --all-questions sweep; defaults to LMMS-Eval's batch size.",
    )
    parser.add_argument(
        "--all-questions", action="store_true",
        help="Run every row in the POPE split in batches, without interactive screens, and save JSONL results.",
    )
    parser.add_argument(
        "--mode", choices=("expert", "amateur", "contrastive", "all"), default="expert",
        help="Model mode for --all-questions and the initial interactive question.",
    )
    parser.add_argument(
        "--capture-point", choices=("block_input", "after_layer_norm", "value"),
        default="block_input", help="Decoder activation used for candidate-token gradient maps.",
    )
    parser.add_argument("--device", default=None)
    for role in ("expert", "amateur"):
        for tier in ("gpu", "cpu"):
            parser.add_argument(
                f"--max-{role}-model-memory-{tier}", type=parse_memory_size, default=None,
                help=f"Optional {role} model {tier.upper()} placement cap; bare numbers mean GB.",
            )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--interactive-timeout-minutes", type=float, default=10.0)
    parser.add_argument("--log-level", choices=("none", "normal", "extensive"), default="normal")
    parser.add_argument("--quiet-warnings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else repository_path(path)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not math.isfinite(args.contrastive_weight) or args.contrastive_weight < 0:
        raise ValueError("--contrastive-weight must be finite and non-negative")
    if not math.isfinite(args.plausibility_threshold) or not 0 < args.plausibility_threshold <= 1:
        raise ValueError("--plausibility-threshold must be in (0, 1]")
    if not math.isfinite(args.interactive_timeout_minutes) or args.interactive_timeout_minutes <= 0:
        raise ValueError("--interactive-timeout-minutes must be finite and positive")
    if args.question_id is not None and args.row_id is not None:
        raise ValueError("Use only one of --question-id or --row-id")
    if args.question_id is not None:
        args.question_id = str(args.question_id).strip()
        if not args.question_id:
            raise ValueError("--question-id cannot be empty")
    if args.row_id is not None:
        args.row_id = str(args.row_id).strip()
        if not args.row_id:
            raise ValueError("--row-id cannot be empty")


def read_session_input(prompt: str, timeout: float) -> str:
    """Read a command, converting EOF and idle timeout into a clean session exit."""
    try:
        return read_timed_input(prompt, timeout)
    except EOFError as error:
        raise InteractiveSessionExit from error
    except InteractiveTimeoutError as error:
        log("Interactive timeout reached; exiting")
        raise InteractiveSessionExit from error


def load_pope_dataset(
    dataset_name: str, split: str
) -> tuple[Any, dict[str, list[int]], dict[str, int], dict[str, list[int]]]:
    """Load POPE metadata and build question/image indexes without decoding all images."""
    from datasets import load_dataset

    log(f"Loading POPE dataset={dataset_name!r}, split={split!r}; first run may download it")
    started = time.monotonic()
    dataset = load_dataset(dataset_name, split=split, token=True)
    columns = set(dataset.column_names)
    required = {"id", "question_id", "question", "answer", "image"}
    missing = required - columns
    if missing:
        raise ValueError(f"POPE dataset is missing required columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError(f"POPE split {split!r} is empty")
    if "image_source" not in columns and "image_id" not in columns:
        raise ValueError("POPE needs an image_source or image_id column to find all questions for an image")

    row_ids = [str(value) for value in dataset["id"]]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("POPE id values must be unique within the selected split")
    row_index = {row_id: index for index, row_id in enumerate(row_ids)}
    qid_values = [str(value) for value in dataset["question_id"]]
    question_index: dict[str, list[int]] = {}
    for index, qid in enumerate(qid_values):
        question_index.setdefault(qid, []).append(index)
    image_column = "image_source" if "image_source" in columns else "image_id"
    image_values = [str(value) for value in dataset[image_column]]
    image_index: dict[str, list[int]] = {}
    for index, image_key in enumerate(image_values):
        image_index.setdefault(image_key, []).append(index)
    log(f"Loaded {len(dataset)} POPE records and indexed {len(image_index)} images in {time.monotonic() - started:.1f}s")
    duplicate_qids = sum(len(positions) > 1 for positions in question_index.values())
    if duplicate_qids:
        log(f"Found {duplicate_qids} question_id values repeated across rows; use row IDs to disambiguate")
    return dataset, question_index, row_index, image_index


def make_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"]).strip()
    return {
        "id": str(row["question_id"]),
        "image": "local",
        "conversations": [{
            "from": "human",
            "value": f"<image>\n{question}\nAnswer the question using a single word or phrase.",
        }, {
            # LLaVA's conversation template needs an empty assistant turn so
            # generation starts after `### Assistant:` rather than after the
            # human turn separator (which can make the model emit EOS at once).
            "from": "gpt",
            "value": "",
        }],
    }


def prepare_position_prompt(adapter: Any, image: Any, prompt_row: dict[str, Any]) -> dict[str, Any]:
    """Prepare multimodal prompt tensors and token labels for position inspection."""
    from investigate_grad_important import decode_non_image_token_labels

    prepared = adapter.preprocess_data([image], prompt_row)
    batch = adapter.data_collator([prepared])
    prompt_inputs, prompt_kwargs = adapter.generate_input(batch)
    image_mask = prompt_kwargs["vision_mask"][0].bool().detach().cpu()
    attention_mask = prompt_kwargs["attention_mask"]
    text_labels = decode_non_image_token_labels(adapter, batch)
    valid_text = attention_mask[0].bool().detach().cpu() & ~image_mask
    if len(text_labels) != int(valid_text.sum()):
        raise RuntimeError("Prompt text labels do not align with non-image decoder positions")
    return {
        "prompt_embeds": prompt_inputs["inputs_embeds"],
        "attention_mask": attention_mask,
        "image_mask": image_mask,
        "text_labels": text_labels,
    }


def generate_lmms_batch(
    model: Any,
    adapter: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    images: list[Any],
    max_new_tokens: int,
    *,
    collect_distributions: bool = False,
) -> tuple[list[list[int]], list[str], list[Any]]:
    """Run ordinary greedy generation with the same LLaVA path as LMMS-Eval."""
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    conv_template = "vicuna_v1"  # LMMS-Eval's Llava model default.
    contexts = [
        f"{str(row['question']).strip()}\nAnswer the question using a single word or phrase."
        for row in rows
    ]
    prompt_texts = []
    for context in contexts:
        question = context
        if DEFAULT_IMAGE_TOKEN not in question:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = (
            copy.deepcopy(conv_templates[conv_template])
            if "llama_3" in conv_template
            else conv_templates[conv_template].copy()
        )
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_texts.append(conv.get_prompt())

    image_tensor = process_images(images, adapter.image_processor, model.config)
    input_device = model.get_input_embeddings().weight.device
    if isinstance(image_tensor, list):
        image_tensor = [image.to(dtype=torch.float16, device=input_device) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device=input_device)

    input_ids_list = [
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for prompt in prompt_texts
    ]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if tokenizer.padding_side == "left":
        input_ids_list = [torch.flip(ids, [0]) for ids in input_ids_list]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    input_ids = input_ids.to(input_device)
    attention_mask = input_ids.ne(pad_token_id).to(input_device)
    image_sizes = [image.size for image in images]

    generation_kwargs = dict(
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    with torch.inference_mode():
        if collect_distributions:
            result = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_kwargs,
            )
            generated_sequences = result.sequences
            step_scores = list(result.scores or ())
        else:
            # Match LMMS-Eval's ordinary path: request the generated token tensor
            # directly, without retaining every vocabulary score tensor.
            generated_sequences = model.generate(input_ids, **generation_kwargs)
            step_scores = []

    if generated_sequences.ndim == 1:
        generated_sequences = generated_sequences.unsqueeze(0)
    if generated_sequences.shape[0] != len(rows):
        if len(rows) > 1:
            # Some LLaVA/Transformers combinations collapse the output batch
            # when multimodal inputs are passed as a list. Retry row by row so
            # a long POPE sweep completes and each answer stays aligned.
            log(
                "WARNING: LLaVA returned %d generations for a batch of %d; "
                "retrying this batch one question at a time"
                % (generated_sequences.shape[0], len(rows))
            )
            singles = [
                generate_lmms_batch(
                    model, adapter, tokenizer, [row], [image], max_new_tokens,
                    collect_distributions=collect_distributions,
                )
                for row, image in zip(rows, images)
            ]
            return (
                [item[0][0] for item in singles],
                [item[1][0] for item in singles],
                [item[2][0] if item[2] else None for item in singles],
            )
        raise RuntimeError(
            "LLaVA returned %d sequences for one POPE question"
            % generated_sequences.shape[0]
        )

    # LLaVA's generate override supplies multimodal inputs_embeds to Transformers;
    # returned sequences contain generated tokens (the same tensor LMMS decodes).
    if step_scores:
        generated_tensor = generated_sequences[:, -len(step_scores):]
    else:
        generated_tensor = generated_sequences
    eos_ids = eos_token_ids(model, tokenizer)
    generated: list[list[int]] = []
    for row_index in range(len(rows)):
        token_ids = []
        for token_id in generated_tensor[row_index].tolist():
            if token_id in eos_ids or token_id == pad_token_id:
                break
            token_ids.append(int(token_id))
        generated.append(token_ids)

    texts = [
        tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        for tokens in generated
    ]
    distributions = [torch.softmax(scores.float(), dim=-1).detach().cpu() for scores in step_scores]
    return generated, texts, distributions


def run_generation(
    mode: str,
    weight: float,
    row: dict[str, Any],
    expert: Any,
    expert_adapter: Any,
    expert_tokenizer: Any,
    amateur: Any,
    amateur_adapter: Any,
    amateur_tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    image = row["image"]
    if image is None or not hasattr(image, "convert"):
        raise ValueError(f"POPE question_id={row['question_id']} has no readable image")
    image = image.convert("RGB")
    prompt_row = make_prompt_row(row)

    if mode == "amateur":
        position_prompt = prepare_position_prompt(amateur_adapter, image, prompt_row)
        prompt_embeds = position_prompt["prompt_embeds"]
        prompt_mask = position_prompt["attention_mask"]
        model, tokenizer = amateur, amateur_tokenizer
        amateur_state = None
        gradient_role = "amateur"
    else:
        position_prompt = prepare_position_prompt(expert_adapter, image, prompt_row)
        prompt_embeds = position_prompt["prompt_embeds"]
        prompt_mask = position_prompt["attention_mask"]
        model, tokenizer = expert, expert_tokenizer
        amateur_state = None
        gradient_role = "expert"
        if mode == "contrastive":
            amateur_prompt = prepare_position_prompt(amateur_adapter, image, prompt_row)
            amateur_embeds, amateur_mask = amateur_prompt["prompt_embeds"], amateur_prompt["attention_mask"]
            amateur_state = (amateur, amateur_embeds, amateur_mask)

    if mode == "contrastive":
        token_ids, answer, distributions = decode_one(
            model, tokenizer, prompt_embeds, prompt_mask, args.max_new_tokens,
            amateur=amateur_state,
            contrastive_weight=weight,
            contrastive_score_type=args.contrastive_score_type,
            plausibility_threshold=args.plausibility_threshold,
            collect_distributions=True,
        )
    else:
        generated, texts, distributions = generate_lmms_batch(
            model, amateur_adapter if mode == "amateur" else expert_adapter,
            tokenizer, [row], [image], args.max_new_tokens,
            collect_distributions=True,
        )
        token_ids, answer = generated[0], texts[0]
    return {
        "mode": mode,
        "weight": weight if mode == "contrastive" else None,
        "answer": answer,
        "prediction": answer.strip().lower(),
        "target": str(row["answer"]).strip().lower(),
        "exact_match_percent": 100 if answer.strip().lower() == str(row["answer"]).strip().lower() else 0,
        "position_inspection": {
            **position_prompt,
            "mode": mode,
            "token_ids": torch.tensor([token_ids], dtype=torch.long, device=prompt_embeds.device),
            "distributions": distributions,
            "adapter": expert_adapter if gradient_role == "expert" else amateur_adapter,
            "gradient_role": gradient_role,
            "question": str(row["question"]),
            "image": image,
        },
    }


def run_generation_batch(
    mode: str,
    weight: float,
    rows: list[dict[str, Any]],
    expert: Any,
    expert_adapter: Any,
    expert_tokenizer: Any,
    amateur: Any,
    amateur_adapter: Any,
    amateur_tokenizer: Any,
    args: argparse.Namespace,
    *,
    include_position_data: bool = True,
) -> list[dict[str, Any]]:
    """Generate a batch with LMMS-Eval's LLaVA path or custom contrastive decoding."""
    import torch

    if not rows:
        return []
    if mode == "amateur":
        adapter, model, tokenizer = amateur_adapter, amateur, amateur_tokenizer
        amateur_inputs = None
        gradient_role = "amateur"
    else:
        adapter, model, tokenizer = expert_adapter, expert, expert_tokenizer
        amateur_inputs = None
        gradient_role = "expert"

    images = []
    for row in rows:
        image = row["image"]
        if image is None or not hasattr(image, "convert"):
            raise ValueError(f"POPE question_id={row['question_id']} has no readable image")
        image = image.convert("RGB")
        images.append(image)

    # Dataset sweeps only need answers. Avoid doing a second multimodal forward
    # through the QIG adapter solely to build interactive token-position data.
    if mode != "contrastive" and not include_position_data:
        generated, texts, _ = generate_lmms_batch(
            model, adapter, tokenizer, rows, images, args.max_new_tokens,
        )
        return [
            {
                "mode": mode,
                "weight": None,
                "answer": answer,
                "prediction": answer.strip().lower(),
                "target": str(row["answer"]).strip().lower(),
                "exact_match_percent": 100 if answer.strip().lower() == str(row["answer"]).strip().lower() else 0,
            }
            for row, answer in zip(rows, texts)
        ]

    prepared = [
        adapter.preprocess_data([image], make_prompt_row(row))
        for image, row in zip(images, rows)
    ]
    batch = adapter.data_collator(prepared)
    prompt_inputs, prompt_kwargs = adapter.generate_input(batch)
    prompt_embeds = prompt_inputs["inputs_embeds"]
    prompt_mask = prompt_kwargs["attention_mask"]

    if mode == "contrastive":
        amateur_prepared = [
            amateur_adapter.preprocess_data([image], make_prompt_row(row))
            for image, row in zip(images, rows)
        ]
        amateur_batch = amateur_adapter.data_collator(amateur_prepared)
        amateur_prompt_inputs, amateur_prompt_kwargs = amateur_adapter.generate_input(amateur_batch)
        amateur_inputs = (
            amateur,
            amateur_prompt_inputs["inputs_embeds"],
            amateur_prompt_kwargs["attention_mask"],
        )

    if mode == "contrastive":
        generated, texts, distributions = decode_batch(
            model, tokenizer, prompt_embeds, prompt_mask, args.max_new_tokens,
            amateur=amateur_inputs,
            contrastive_weight=weight,
            contrastive_score_type=args.contrastive_score_type,
            plausibility_threshold=args.plausibility_threshold,
        )
    else:
        generated, texts, distributions = generate_lmms_batch(
            model, adapter, tokenizer, rows, images, args.max_new_tokens,
        )

    if not include_position_data:
        return [
            {
                "mode": mode,
                "weight": weight if mode == "contrastive" else None,
                "answer": answer,
                "prediction": answer.strip().lower(),
                "target": str(row["answer"]).strip().lower(),
                "exact_match_percent": 100 if answer.strip().lower() == str(row["answer"]).strip().lower() else 0,
            }
            for row, answer in zip(rows, texts)
        ]

    # Keep the existing position-inspection screen available for the resulting answers.
    from investigate_grad_important import decode_non_image_token_labels

    outputs = []
    image_mask = prompt_kwargs["vision_mask"]
    for index, (row, image, answer) in enumerate(zip(rows, images, texts)):
        valid = prompt_mask[index].bool()
        # Remove collator padding so the single-answer position inspector sees its usual input shape.
        sample_embeds = prompt_embeds[index:index + 1, valid.to(prompt_embeds.device)]
        sample_image_mask = image_mask[index, valid.to(image_mask.device)].detach().cpu()
        sample_labels = decode_non_image_token_labels(adapter, adapter.data_collator([prepared[index]]))
        sample_distributions = [scores[index:index + 1] for scores in distributions[:len(generated[index])]]
        output = {
            "mode": mode,
            "weight": weight if mode == "contrastive" else None,
            "answer": answer,
            "prediction": answer.strip().lower(),
            "target": str(row["answer"]).strip().lower(),
            "exact_match_percent": 100 if answer.strip().lower() == str(row["answer"]).strip().lower() else 0,
            "position_inspection": {
                "prompt_embeds": sample_embeds,
                "attention_mask": torch.ones((1, sample_embeds.shape[1]), dtype=prompt_mask.dtype,
                                              device=prompt_mask.device),
                "image_mask": sample_image_mask,
                "text_labels": sample_labels,
                "mode": mode,
                "token_ids": torch.tensor([generated[index]], dtype=torch.long, device=sample_embeds.device),
                "distributions": sample_distributions,
                "adapter": adapter,
                "gradient_role": gradient_role,
                "question": str(row["question"]),
                "image": image,
            },
        }
        outputs.append(output)
    return outputs


def decode_batch(
    model: Any,
    tokenizer: Any,
    prompt_embeds: Any,
    prompt_mask: Any,
    max_new_tokens: int,
    *,
    amateur: tuple[Any, Any, Any] | None = None,
    contrastive_weight: float = 1.0,
    contrastive_score_type: str = "logits",
    plausibility_threshold: float = 0.001,
) -> tuple[list[list[int]], list[str], list[Any]]:
    """Greedy autoregressive decoding for a batch, retaining per-row KV caches."""
    import torch

    batch_size = prompt_embeds.shape[0]
    embedding = model.get_input_embeddings()
    current_embeds = prompt_embeds
    attention = prompt_mask
    past = None
    amateur_model, amateur_current, amateur_attention = amateur or (None, None, None)
    amateur_past = None
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    distributions: list[Any] = []
    token_device = embedding.weight.device
    finished = torch.zeros(batch_size, dtype=torch.bool, device=token_device)
    eos_ids = eos_token_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id or next(iter(eos_ids), 0)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            expert_outputs = model(
                inputs_embeds=current_embeds, attention_mask=attention, past_key_values=past,
                use_cache=True, return_dict=True,
            )
            logits = expert_outputs.logits.float()
            if past is None:
                logits_mask = attention.to(logits.device).bool()
                positions = torch.arange(logits_mask.shape[1], device=logits.device).expand_as(logits_mask)
                last_positions = positions.masked_fill(~logits_mask, -1).max(dim=1).values
                scores = logits[torch.arange(batch_size, device=logits.device), last_positions]
            else:
                scores = logits[:, -1, :]
            scores = scores.to(token_device)

            if amateur_model is not None:
                amateur_outputs = amateur_model(
                    inputs_embeds=amateur_current, attention_mask=amateur_attention,
                    past_key_values=amateur_past, use_cache=True, return_dict=True,
                )
                amateur_logits_all = amateur_outputs.logits.float()
                if amateur_past is None:
                    amateur_mask = amateur_attention.to(amateur_logits_all.device).bool()
                    amateur_positions = torch.arange(amateur_mask.shape[1], device=amateur_logits_all.device).expand_as(amateur_mask)
                    amateur_last = amateur_positions.masked_fill(~amateur_mask, -1).max(dim=1).values
                    amateur_logits = amateur_logits_all[
                        torch.arange(batch_size, device=amateur_logits_all.device), amateur_last
                    ].to(token_device)
                else:
                    amateur_logits = amateur_logits_all[:, -1, :].to(token_device)
                if scores.shape[-1] != amateur_logits.shape[-1]:
                    raise RuntimeError("Expert and amateur output vocabulary sizes differ")
                expert_log_probs = torch.log_softmax(scores, dim=-1)
                allowed = expert_log_probs.exp() >= expert_log_probs.exp().amax(dim=-1, keepdim=True) * plausibility_threshold
                if contrastive_score_type == "logits":
                    expert_scores, amateur_scores = scores, amateur_logits
                elif contrastive_score_type == "log-probabilities":
                    expert_scores, amateur_scores = expert_log_probs, torch.log_softmax(amateur_logits, dim=-1)
                else:
                    raise ValueError(f"Unsupported contrastive score type: {contrastive_score_type}")
                scores = ((1 + contrastive_weight) * expert_scores - contrastive_weight * amateur_scores).masked_fill(~allowed, -torch.inf)

            distributions.append(torch.softmax(scores, dim=-1).detach().cpu())
            next_ids = scores.argmax(dim=-1)
            if eos_ids:
                was_finished = finished.clone()
                for eos_id in eos_ids:
                    finished |= next_ids.eq(eos_id)
                next_ids = torch.where(was_finished, torch.full_like(next_ids, pad_id), next_ids)
            else:
                was_finished = finished.clone()
            for idx, token_id in enumerate(next_ids.tolist()):
                if not was_finished[idx]:
                    generated[idx].append(int(token_id))
            if bool(finished.all()):
                break

            active = ~finished
            past = expert_outputs.past_key_values
            token_embeds = embedding(next_ids[:, None].to(embedding.weight.device))
            current_embeds = token_embeds
            attention = torch.cat(
                (attention, active[:, None].to(device=attention.device, dtype=attention.dtype)), dim=1
            )
            if amateur_model is not None:
                amateur_past = amateur_outputs.past_key_values
                amateur_embedding = amateur_model.get_input_embeddings()
                amateur_current = amateur_embedding(next_ids[:, None].to(amateur_embedding.weight.device))
                amateur_attention = torch.cat(
                    (amateur_attention, active[:, None].to(amateur_attention.device, amateur_attention.dtype)), dim=1
                )

    texts = [tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for tokens in generated]
    return generated, texts, distributions


def wrap_column(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def show_comparison_screen(row: dict[str, Any], outputs: list[dict[str, Any]], timeout: float) -> str:
    """Display the three model answers side by side as a separate screen."""
    width = max(24, min(48, (shutil.get_terminal_size(fallback=(120, 24)).columns - 8) // 3))
    columns = [
        (f"EXPERT {outputs[0]['exact_match_percent']}%", outputs[0]["answer"]),
        (f"AMATEUR {outputs[1]['exact_match_percent']}%", outputs[1]["answer"]),
        (f"CONTRASTIVE w={outputs[2]['weight']:g} {outputs[2]['exact_match_percent']}%", outputs[2]["answer"]),
    ]
    print("\n" + "=" * (width * 3 + 8))
    print(f"POPE comparison | row_id={row['id']} | question_id={row['question_id']} | target={row['answer']} | image={row.get('image_source', '')}")
    print(str(row["question"]))
    print("=" * (width * 3 + 8))
    print(" | ".join(label[:width].ljust(width) for label, _ in columns))
    print("-+-".join("-" * width for _ in columns))
    wrapped = [wrap_column(answer, width) for _, answer in columns]
    for line in range(max(map(len, wrapped))):
        print(" | ".join((parts[line] if line < len(parts) else "").ljust(width) for parts in wrapped))
    while True:
        try:
            command = read_timed_input("Comparison screen: b back, q quit: ", timeout).strip().lower()
        except EOFError:
            return "quit"
        except InteractiveTimeoutError:
            log("Interactive timeout reached; exiting")
            return "quit"
        if command in {"b", "back", ""}:
            return "back"
        if command in {"q", "quit"}:
            return "quit"
        print("Use b to return or q to exit.")


def inspect_answer_positions(
    output: dict[str, Any],
    capture_point: str,
    timeout: float,
) -> str:
    """Open the gradient investigator's token-position and candidate screens read-only."""
    from investigate_grad_important import (
        activation_target,
        generate_candidate_continuation,
        score_candidate_token_gradients,
        show_gradient_heatmaps,
        show_large_source_image,
    )
    from attention_quantization.interactive_terminal import inspect_generated_token_positions

    inspection = output["position_inspection"]
    role = inspection["gradient_role"]
    print(
        f"Inspecting {inspection['mode']} answer positions; token candidates use its generation scores, "
        f"candidate gradient maps use the {role} model."
    )
    capture_target = activation_target(capture_point)
    try:
        action = inspect_generated_token_positions(
            inspection["adapter"],
            capture_target,
            inspection["prompt_embeds"],
            inspection["attention_mask"],
            inspection["image_mask"],
            inspection["text_labels"],
            inspection["token_ids"],
            inspection["distributions"],
            len(inspection["token_ids"][0]),
            inspection["question"],
            inspection["image"],
            1,
            heatmap_only=False,
            scale="log",
            score_normalization="raw",
            top_token_percent=None,
            top_token_cumulative_percent=None,
            score_candidate_token_gradients_fn=score_candidate_token_gradients,
            generate_candidate_continuation_fn=generate_candidate_continuation,
            show_gradient_heatmaps_fn=show_gradient_heatmaps,
            show_large_source_image_fn=show_large_source_image,
            log_fn=log,
            interactive_timeout_seconds=timeout,
            read_only_inspection=True,
        )
    except InteractiveTimeoutError:
        log("Position inspection timed out; returning to the POPE screen")
        return "back"
    if action and action["action"] == "quit":
        return "quit"
    return "back"


def result_record(row: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    """Strip live tensors and image objects before writing a result line."""
    return {
        "timestamp": datetime.now().isoformat(),
        "id": str(row["id"]),
        "question_id": str(row["question_id"]),
        "image_source": row.get("image_source"),
        "question": row["question"],
        "target": output["target"],
        "mode": output["mode"],
        "weight": output["weight"],
        "answer": output["answer"],
        "prediction": output["prediction"],
        "exact_match_percent": output["exact_match_percent"],
    }


def render_main_screen(row: dict[str, Any], selected_mode: str, weight: float,
                       output: dict[str, Any] | None, index: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"POPE question {index}/{total} | row_id={row['id']} | question_id={row['question_id']} | image={row.get('image_source', '')}")
    print(f"Category: {row.get('category', 'unknown')} | Target: {row['answer']}")
    print(f"Question: {row['question']}")
    print(f"Selected model mode: {selected_mode}" + (f" (weight={weight:g})" if selected_mode == "contrastive" else ""))
    if output is None:
        print("Answer: (not generated)")
    elif output.get("mode") == "all":
        print("Answers: shown on the comparison screen")
    else:
        print(f"Answer: {output['answer']!r} | POPE exact match: {output['exact_match_percent']}%")
    print("Commands: m [expert|amateur|contrastive [weight]|all], s [question_id|row:ID|random|all], p [model] positions, i image, h help, q quit")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, quiet_warnings=args.quiet_warnings)
    validate_args(args)
    # ========================================================================
    # 1. PREPARE POPE DATASET AND QUESTION INDEXES
    # ========================================================================
    dataset, question_index, row_index, image_index = load_pope_dataset(args.dataset, args.split)
    rng = random.Random(args.seed)
    if args.row_id is not None:
        if args.row_id not in row_index:
            raise ValueError(f"POPE row id {args.row_id!r} was not found in {args.dataset}/{args.split}")
        current_index = row_index[args.row_id]
    elif args.question_id is not None:
        if args.question_id not in question_index:
            raise ValueError(f"POPE question_id {args.question_id!r} was not found in {args.dataset}/{args.split}")
        matches = question_index[args.question_id]
        current_index = matches[0]
        if len(matches) > 1:
            matching_rows = [dict(dataset[position]) for position in matches]
            print(f"question_id={args.question_id} matches {len(matches)} rows; starting with the first:")
            for row in matching_rows:
                print(f"  row_id={row['id']} category={row.get('category', 'unknown')} image={row.get('image_source', '')}")
    else:
        current_index = rng.randrange(len(dataset))

    # ========================================================================
    # 2. LOAD MODEL(S)
    # ========================================================================
    model_config = read_yaml(args.model_config)
    expert_checkpoint = resolve_path(args.expert_model or model_config.get("model_dir") or model_config.get("model_id"))
    amateur_checkpoint = resolve_path(args.amateur_model) if args.amateur_model else None
    if args.mode in {"amateur", "contrastive", "all"} and amateur_checkpoint is None:
        raise ValueError(f"--mode {args.mode} requires --amateur-model")
    device = args.device or model_config.get("device_map", "cuda:0")
    if device not in {"cpu", "cuda", "auto"} and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("--device must be cpu, auto, cuda, or cuda:N")
    args.device = device
    args.timing = args.log_level == "extensive"
    output_dir = resolve_path(args.output_dir or Path("outputs/investigations") / "pope-interactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / ("all_questions_results.jsonl" if args.all_questions else "interactive_results.jsonl")
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"Results already exist: {results_path}; pass --overwrite to replace them")
    qig_root = resolve_path(model_config.get("implementation_source_dir", "third_party/QIG"))
    builder, get_process_model = load_runtime(qig_root)
    log(f"Loading expert={expert_checkpoint}; device={device}")
    expert, expert_adapter, expert_tokenizer = load_model(
        builder, get_process_model, expert_checkpoint, model_config, args, "expert", device, output_dir,
    )
    amateur = amateur_adapter = amateur_tokenizer = None
    if amateur_checkpoint is not None:
        log(f"Loading amateur={amateur_checkpoint}; device={device}")
        amateur, amateur_adapter, amateur_tokenizer = load_model(
            builder, get_process_model, amateur_checkpoint, model_config, args, "amateur", device, output_dir,
        )
        validate_compatibility(expert_tokenizer, amateur_tokenizer, expert_adapter, amateur_adapter)
        log("Expert and amateur tokenizer/image processing settings match", level="extensive")

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ========================================================================
    # 3. RUN GENERATION AND CAPTURE CANDIDATE GRADIENTS
    # ========================================================================
    log(
        "Expert/amateur generation uses LMMS-Eval LLaVA greedy generation; "
        "contrastive mode uses the custom logits decoder",
    )
    log("POPE generation uses inference mode without capture hooks", level="extensive")

    timeout = args.interactive_timeout_minutes * 60
    selected_mode = args.mode
    selected_weight = args.contrastive_weight
    current_output: dict[str, Any] | None = None
    result_file = results_path.open("w", encoding="utf-8")
    def row_at(position: int) -> dict[str, Any]:
        return dict(dataset[position])

    def generate_for_current(row: dict[str, Any], mode: str, weight: float) -> dict[str, Any]:
        if mode in {"amateur", "contrastive", "all"} and amateur is None:
            raise ValueError("This mode requires --amateur-model")
        if mode == "all":
            outputs = [
                run_generation("expert", weight, row, expert, expert_adapter, expert_tokenizer,
                               amateur, amateur_adapter, amateur_tokenizer, args),
                run_generation("amateur", weight, row, expert, expert_adapter, expert_tokenizer,
                               amateur, amateur_adapter, amateur_tokenizer, args),
                run_generation("contrastive", weight, row, expert, expert_adapter, expert_tokenizer,
                               amateur, amateur_adapter, amateur_tokenizer, args),
            ]
            for output in outputs:
                result_file.write(json.dumps(result_record(row, output), ensure_ascii=False) + "\n")
            result_file.flush()
            screen_action = show_comparison_screen(row, outputs, timeout)
            if screen_action == "quit":
                raise InteractiveSessionExit
            return {"mode": "all", "outputs": outputs, "target": str(row["answer"]).strip().lower()}
        output = run_generation(mode, weight, row, expert, expert_adapter, expert_tokenizer,
                                amateur, amateur_adapter, amateur_tokenizer, args)
        result_file.write(json.dumps(result_record(row, output), ensure_ascii=False) + "\n")
        result_file.flush()
        return output

    def generate_all_for_rows(
        rows: list[dict[str, Any]], mode: str, weight: float, *, interactive: bool = True
    ) -> list[dict[str, Any]]:
        """Generate a batch of questions and save each answer."""
        if mode in {"amateur", "contrastive", "all"} and amateur is None:
            raise ValueError("This mode requires --amateur-model")
        modes = ("expert", "amateur", "contrastive") if mode == "all" else (mode,)
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for one_mode in modes:
            by_mode[one_mode] = run_generation_batch(
                one_mode, weight, rows, expert, expert_adapter, expert_tokenizer,
                amateur, amateur_adapter, amateur_tokenizer, args,
                include_position_data=interactive,
            )
        results = []
        for row_idx, row in enumerate(rows):
            row_outputs = [by_mode[one_mode][row_idx] for one_mode in modes]
            if mode == "all":
                for output in row_outputs:
                    result_file.write(json.dumps(result_record(row, output), ensure_ascii=False) + "\n")
                result = {"mode": "all", "outputs": row_outputs, "target": str(row["answer"]).strip().lower()}
                if interactive and show_comparison_screen(row, row_outputs, timeout) == "quit":
                    result_file.flush()
                    raise InteractiveSessionExit
            else:
                result = row_outputs[0]
                result_file.write(json.dumps(result_record(row, result), ensure_ascii=False) + "\n")
                if interactive:
                    print(
                        f"Answer: {result['answer']!r} | target={result['target']} "
                        f"| POPE exact match: {result['exact_match_percent']}%"
                    )
            results.append(result)
        result_file.flush()
        return results

    try:
        if args.all_questions:
            batch_count = math.ceil(len(dataset) / args.batch_size)
            log(
                f"Running all {len(dataset)} POPE rows in {batch_count} batches "
                f"(batch_size={args.batch_size}, mode={selected_mode}, weight={selected_weight:g}); "
                f"writing {results_path}"
            )
            progress_interval = max(1, math.ceil(batch_count / 100))
            for offset in range(0, len(dataset), args.batch_size):
                rows = [row_at(index) for index in range(offset, min(offset + args.batch_size, len(dataset)))]
                batch_number = offset // args.batch_size + 1
                if batch_number == 1 or batch_number % progress_interval == 0 or batch_number == batch_count:
                    log(f"POPE batch {batch_number}/{batch_count}: rows {offset}-{offset + len(rows) - 1}")
                generate_all_for_rows(rows, selected_mode, selected_weight, interactive=False)
            log(f"Finished all {len(dataset)} POPE rows; results saved to {results_path}")
            return 0

        current_row = row_at(current_index)
        current_output = generate_for_current(current_row, selected_mode, selected_weight)
        while True:
            current_row = row_at(current_index)
            render_main_screen(current_row, selected_mode, selected_weight, current_output,
                               current_index + 1, len(dataset))
            try:
                parts = read_session_input("POPE> ", timeout).strip().split()
            except InteractiveSessionExit:
                break
            if not parts:
                continue
            command = parts[0].lower()
            if command in {"q", "quit", "exit"}:
                break
            if command in {"h", "help"}:
                print("m expert | m amateur | m contrastive [weight] | m all")
                print(f"s ID | s row:ID | s random | s all (same-image questions, batches of {args.batch_size})")
                print("p [expert|amateur|contrastive] opens token-position and candidate-gradient screens")
                print("i displays the current image; q exits. The comparison screen uses b to return.")
                continue
            if command == "i" and len(parts) == 1:
                import matplotlib.pyplot as plt
                image = current_row["image"]
                plt.imshow(image.convert("RGB"))
                plt.axis("off")
                plt.title(f"POPE question_id={current_row['question_id']}")
                plt.show()
                continue
            if command == "p":
                if selected_mode == "all":
                    if current_output is None or current_output.get("mode") != "all":
                        print("Run m all first to create the three answers for this question.")
                        continue
                    if len(parts) < 2:
                        choice = read_session_input(
                            "Inspect positions for [expert/amateur/contrastive]: ", timeout
                        ).strip().lower()
                    else:
                        choice = parts[1].lower()
                    outputs_by_mode = {item["mode"]: item for item in current_output["outputs"]}
                    if choice not in outputs_by_mode or len(parts) > 2:
                        print("Use p expert, p amateur, or p contrastive.")
                        continue
                    selected_output = outputs_by_mode[choice]
                else:
                    if len(parts) > 1:
                        print("Use p to inspect the current answer's token positions.")
                        continue
                    if current_output is None:
                        print("Generate an answer first.")
                        continue
                    selected_output = current_output
                position_action = inspect_answer_positions(selected_output, args.capture_point, timeout)
                if position_action == "quit":
                    raise InteractiveSessionExit
                continue
            if command == "m":
                if len(parts) < 2:
                    choice = read_session_input(
                        "Model [expert/amateur/contrastive/all]: ", timeout
                    ).strip().lower().split()
                    parts = ["m", *choice]
                if len(parts) not in {2, 3} or (len(parts) == 3 and parts[1].lower() != "contrastive"):
                    print("Use m expert, m amateur, m contrastive [weight], or m all.")
                    continue
                mode = parts[1].lower()
                if mode not in {"expert", "amateur", "contrastive", "all"}:
                    print("Choose expert, amateur, contrastive, or all.")
                    continue
                if mode in {"amateur", "contrastive", "all"} and amateur is None:
                    print("Load an amateur checkpoint with --amateur-model to use that mode.")
                    continue
                if mode == "contrastive" and len(parts) == 3:
                    try:
                        candidate_weight = float(parts[2])
                    except ValueError:
                        print("Contrastive weight must be a number.")
                        continue
                    if not math.isfinite(candidate_weight) or candidate_weight < 0:
                        print("Contrastive weight must be finite and non-negative.")
                        continue
                    selected_weight = candidate_weight
                selected_mode = mode
                current_output = generate_for_current(current_row, selected_mode, selected_weight)
                continue
            if command == "s":
                if len(parts) < 2:
                    parts = ["s", read_session_input("Question ID, random, or all: ", timeout).strip()]
                selection = " ".join(parts[1:]).strip()
                if selection.lower() == "random":
                    current_index = rng.randrange(len(dataset))
                    current_row = row_at(current_index)
                    current_output = generate_for_current(current_row, selected_mode, selected_weight)
                    continue
                if selection.lower() == "all":
                    source = str(current_row.get("image_source", current_row.get("image_id", "")))
                    positions = image_index.get(source, [])
                    if not positions:
                        print("Could not find questions for this image.")
                        continue
                    image_rows = [row_at(position) for position in positions]
                    batch_count = math.ceil(len(image_rows) / args.batch_size)
                    print(
                        f"Generating {len(image_rows)} questions for image {source} "
                        f"in {batch_count} batch(es) of up to {args.batch_size}"
                    )
                    all_results = []
                    for offset in range(0, len(image_rows), args.batch_size):
                        chunk = image_rows[offset:offset + args.batch_size]
                        print(f"Batch {offset // args.batch_size + 1}/{batch_count}")
                        all_results.extend(generate_all_for_rows(chunk, selected_mode, selected_weight))
                    if all_results:
                        current_index = positions[-1]
                        current_row = row_at(current_index)
                        current_output = all_results[-1]
                    continue
                if selection.lower().startswith("row:"):
                    selected_row_id = selection[4:]
                    if selected_row_id not in row_index:
                        print(f"Row ID {selected_row_id!r} was not found.")
                        continue
                    current_index = row_index[selected_row_id]
                    current_row = row_at(current_index)
                    current_output = generate_for_current(current_row, selected_mode, selected_weight)
                    continue
                if selection not in question_index:
                    print(f"Question ID {selection!r} was not found.")
                    continue
                matches = question_index[selection]
                if len(matches) > 1:
                    print(f"question_id={selection} matches multiple rows; select one with s row:<id>:")
                    for position in matches:
                        matching_row = row_at(position)
                        print(f"  row_id={matching_row['id']} category={matching_row.get('category', 'unknown')} image={matching_row.get('image_source', '')}")
                    continue
                current_index = matches[0]
                current_row = row_at(current_index)
                current_output = generate_for_current(current_row, selected_mode, selected_weight)
                continue
            print("Unknown command. Enter h for help.")
    except InteractiveSessionExit:
        pass
    finally:
        result_file.close()
    # ========================================================================
    # 4. CLOSE AND REPORT SAVED RESPONSES
    # ========================================================================
    print(f"Interactive POPE responses saved to {results_path}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except InteractiveSessionExit:
        print("Interactive POPE session ended.")
        exit_code = 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        exit_code = 1
    if exit_code and "IPython" not in sys.modules:
        raise SystemExit(exit_code)
