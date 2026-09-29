"""Create a local ShareGPT4V dataset subset; COCO is the default source.

Examples:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --source sam
    python scripts/prepare_dataset.py --source coco --samples 128 --seed 42

The full selected subset is saved by default. Sampling is optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from attention_quantization.data import (  # noqa: E402
    get_data_paths,
    load_sharegpt4v_dataset,
)


SOURCE_PREFIXES = {
    "coco": ("coco/",),
    "gqa": ("gqa/",),
    "textvqa": ("textvqa/",),
    "visual-genome": ("vg/",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter ShareGPT4V records and save a local dataset; COCO is the default."
    )
    parser.add_argument(
        "--source",
        choices=(*SOURCE_PREFIXES, "all"),
        default="coco",
        help="Image source to keep (default: coco).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "dataset.yaml",
        help="YAML dataset path config (default: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <configured processed_dir>/sharegpt4v_<source>).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional number of records to select. Omit to save the full selected source.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used only with --samples (default: 42)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_data_paths(args.config)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else paths["processed_dir"] / f"sharegpt4v_{args.source}"
    )

    try:
        if args.samples is not None and args.samples < 1:
            raise ValueError("--samples must be a positive integer")

        selected_source = None if args.source == "all" else args.source
        dataset = load_sharegpt4v_dataset(
            source=selected_source,
            config_path=args.config,
            download_images=selected_source is not None,
        )
        if not len(dataset):
            raise ValueError(f"No records found for source {args.source!r}")

        if args.samples is not None:
            if args.samples > len(dataset):
                raise ValueError(
                    f"Requested {args.samples} records, but only {len(dataset)} are available."
                )
            dataset = dataset.shuffle(seed=args.seed).select(range(args.samples))

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))
        print(f"Saved {len(dataset):,} {args.source} records to: {output_dir}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
