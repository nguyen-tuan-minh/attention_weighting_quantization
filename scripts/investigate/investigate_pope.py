"""Interactively inspect POPE questions with LLaVA and optional contrastive decoding."""

from __future__ import annotations

import argparse
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
    load_model,
    load_runtime,
    parse_memory_size,
    prepare_prompt,
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
    parser.add_argument("--max-new-tokens", type=int, default=16)
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
        }],
    }


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
    image = row["image"]
    if image is None or not hasattr(image, "convert"):
        raise ValueError(f"POPE question_id={row['question_id']} has no readable image")
    image = image.convert("RGB")
    prompt_row = make_prompt_row(row)

    if mode == "amateur":
        prompt_embeds, prompt_mask, _ = prepare_prompt(amateur_adapter, image, prompt_row)
        model, tokenizer = amateur, amateur_tokenizer
        amateur_state = None
    else:
        prompt_embeds, prompt_mask, _ = prepare_prompt(expert_adapter, image, prompt_row)
        model, tokenizer = expert, expert_tokenizer
        amateur_state = None
        if mode == "contrastive":
            amateur_embeds, amateur_mask, _ = prepare_prompt(amateur_adapter, image, prompt_row)
            amateur_state = (amateur, amateur_embeds, amateur_mask)

    _, answer = decode_one(
        model, tokenizer, prompt_embeds, prompt_mask, args.max_new_tokens,
        amateur=amateur_state,
        contrastive_weight=weight,
        contrastive_score_type=args.contrastive_score_type,
        plausibility_threshold=args.plausibility_threshold,
    )
    return {
        "mode": mode,
        "weight": weight if mode == "contrastive" else None,
        "answer": answer,
        "prediction": answer.strip().lower(),
        "target": str(row["answer"]).strip().lower(),
    }


def wrap_column(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def show_comparison_screen(row: dict[str, Any], outputs: list[dict[str, Any]], timeout: float) -> str:
    """Display the three model answers side by side as a separate screen."""
    width = max(24, min(48, (shutil.get_terminal_size(fallback=(120, 24)).columns - 8) // 3))
    columns = [("EXPERT", outputs[0]["answer"]), ("AMATEUR", outputs[1]["answer"]),
               (f"CONTRASTIVE w={outputs[2]['weight']:g}", outputs[2]["answer"])]
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
        print(f"Answer: {output['answer']!r} | exact POPE match: {output['prediction'] == output['target']}")
    print("Commands: m [expert|amateur|contrastive [weight]|all], s [question_id|row:ID|random|all], i image, h help, q quit")


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
    device = args.device or model_config.get("device_map", "cuda:0")
    if device not in {"cpu", "cuda", "auto"} and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("--device must be cpu, auto, cuda, or cuda:N")
    args.device = device
    args.timing = args.log_level == "extensive"
    output_dir = resolve_path(args.output_dir or Path("outputs/investigations") / "pope-interactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "interactive_results.jsonl"
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
    # 3. PREPARE INFERENCE SESSION
    # ========================================================================
    log("POPE generation uses inference mode without capture hooks", level="extensive")

    # ========================================================================
    # 4. RUN THE INTERACTIVE QUESTION SESSION
    # ========================================================================
    timeout = args.interactive_timeout_minutes * 60
    selected_mode = "expert"
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
                result_file.write(json.dumps({
                    "timestamp": datetime.now().isoformat(), "id": str(row["id"]),
                    "question_id": str(row["question_id"]),
                    "image_source": row.get("image_source"), "question": row["question"],
                    "target": str(row["answer"]).strip().lower(), **output,
                }, ensure_ascii=False) + "\n")
            result_file.flush()
            screen_action = show_comparison_screen(row, outputs, timeout)
            if screen_action == "quit":
                raise InteractiveSessionExit
            return {"mode": "all", "answer": " | ".join(output["answer"] for output in outputs),
                    "prediction": "", "target": str(row["answer"]).strip().lower()}
        output = run_generation(mode, weight, row, expert, expert_adapter, expert_tokenizer,
                                amateur, amateur_adapter, amateur_tokenizer, args)
        result_file.write(json.dumps({
            "timestamp": datetime.now().isoformat(), "id": str(row["id"]),
            "question_id": str(row["question_id"]),
            "image_source": row.get("image_source"), "question": row["question"],
            "target": str(row["answer"]).strip().lower(), **output,
        }, ensure_ascii=False) + "\n")
        result_file.flush()
        return output

    try:
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
                print("s ID | s row:ID | s random | s all (questions sharing this image, sequentially)")
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
                    for number, position in enumerate(positions, start=1):
                        current_index = position
                        current_row = row_at(current_index)
                        print(f"\n--- Image question {number}/{len(positions)} ---")
                        render_main_screen(current_row, selected_mode, selected_weight, None,
                                           current_index + 1, len(dataset))
                        current_output = generate_for_current(current_row, selected_mode, selected_weight)
                        if selected_mode != "all":
                            print(f"Answer: {current_output['answer']!r} | target={current_output['target']}")
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
    # 5. CLOSE AND REPORT SAVED RESPONSES
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
