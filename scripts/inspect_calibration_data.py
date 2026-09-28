"""Inspect a deterministic ShareGPT4V COCO calibration sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.data import make_calibration_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a reproducibly sampled ShareGPT4V COCO calibration set."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Repository data directory (default: <repository>/data).",
    )
    parser.add_argument(
        "--samples", type=int, default=128, help="Calibration sample count (default: 128)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic sampling seed (default: 42)."
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of selected records to print (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preview < 0:
        print("Error: --preview must be zero or greater.", file=sys.stderr)
        return 2

    try:
        dataset = make_calibration_dataset(
            args.data_dir,
            sample_count=args.samples,
            seed=args.seed,
        )
        print(f"Calibration records selected: {len(dataset)} (seed={args.seed})")
        for index in range(min(args.preview, len(dataset))):
            sample = dataset[index]
            print(f"\n[{index}] record_id: {sample['record_id']}")
            print(f"Image(s): {len(sample['images'])}")
            print(f"User prompt: {sample['user_prompt']}")
            print(f"Assistant caption: {sample['assistant_caption']}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
