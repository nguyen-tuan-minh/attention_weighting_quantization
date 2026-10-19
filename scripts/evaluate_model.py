"""Evaluate an LVLM with the QIG companion LMMS-Eval installation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
    parser.add_argument("--model", default="llava", help="LMMS-Eval model adapter name (default: llava).")
    parser.add_argument("--model-args", default="", help="Additional comma-separated LMMS-Eval model arguments.")
    parser.add_argument("--tasks", default=PAPER_TASKS, help=f"Comma-separated LMMS-Eval task IDs (default: {PAPER_TASKS}).")
    parser.add_argument("--batch-size", default="1", help="LMMS-Eval batch size; use 1 for LLaVA 1.5 compatibility.")
    parser.add_argument("--device", default=None, help="Evaluation device, for example cuda:0.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Result directory (default: outputs/evaluations/<timestamp>).")
    parser.add_argument("--limit", type=float, default=None, help="Limit examples per task for a smoke run; do not use for reported metrics.")
    parser.add_argument("--seed", default="42", help="LMMS-Eval seed (single integer or four comma-separated seeds).")
    parser.add_argument("--log-samples", action="store_true", help="Save per-example prompts, predictions, and targets.")
    parser.add_argument("--verbosity", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
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

    output_dir = args.output_dir or Path("outputs/evaluations") / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = repository_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = [f"pretrained={pretrained}"]
    for key in ("device_map", "attn_implementation"):
        value = model_config.get(key)
        if value is not None:
            model_args.append(f"{key}={value}")
    if args.model_args:
        model_args.append(args.model_args.strip().strip(","))

    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        args.model,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--verbosity",
        args.verbosity,
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.log_samples:
        command.append("--log_samples")

    print(f"Model: {pretrained}", flush=True)
    print(f"Tasks: {args.tasks}", flush=True)
    print(f"LMMS-Eval results: {output_dir}", flush=True)
    print("LMMS-Eval reports each task's configured metric; metrics differ by benchmark.", flush=True)
    try:
        return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode
    except KeyboardInterrupt:
        print("Evaluation interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
