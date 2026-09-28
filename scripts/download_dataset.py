"""Download COCO train2017 and prepare ShareGPT4V COCO calibration records.

Run from any working directory with:
    python scripts/download_dataset.py

The default data directory is <repository>/data. No dataset is downloaded when
this file is imported; downloads start only when the script is run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
EXPECTED_COCO_TRAIN_IMAGES = 118_287
HF_DATASET_ID = "Lin-Chen/ShareGPT4V"
HF_CONFIG = "ShareGPT4V"


def download_file(url: str, destination: Path) -> None:
    """Download a file with a simple progress indicator."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")

    def report_progress(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rDownloading: {percent:5.1f}%", end="", flush=True)
        else:
            print(f"\rDownloaded {downloaded / (1024 * 1024):.1f} MiB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, temporary_path, reporthook=report_progress)
        print()
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_coco_archive(archive_path: Path, destination: Path) -> None:
    """Extract the official archive while rejecting paths outside destination."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in COCO archive: {member.filename}")
        archive.extractall(destination)


def ensure_coco_images(data_root: Path) -> Path:
    coco_root = data_root / "raw" / "coco"
    image_dir = coco_root / "train2017"
    archive_path = coco_root / "train2017.zip"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_count = sum(1 for path in image_dir.glob("*.jpg"))
    if image_count == EXPECTED_COCO_TRAIN_IMAGES:
        print(f"COCO train2017 is already available ({image_count:,} images): {image_dir}")
        return image_dir

    if image_count:
        print(f"Found {image_count:,} COCO images; checking/extracting the full archive.")
    else:
        print("COCO train2017 images not found.")

    if not archive_path.exists():
        print(f"Downloading COCO train2017 archive to: {archive_path}")
        download_file(COCO_TRAIN2017_URL, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    print(f"Extracting COCO images into: {coco_root}")
    extract_coco_archive(archive_path, coco_root)
    image_count = sum(1 for path in image_dir.glob("*.jpg"))
    if image_count != EXPECTED_COCO_TRAIN_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_COCO_TRAIN_IMAGES:,} COCO train images, found {image_count:,} "
            f"in {image_dir}. Check the archive and available disk space."
        )
    print(f"COCO train2017 ready ({image_count:,} images): {image_dir}")
    return image_dir


def prepare_calibration_manifest(
    data_root: Path, *, sample_count: int, seed: int
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: "
            "python -m pip install datasets"
        ) from error

    cache_dir = data_root / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {HF_DATASET_ID} ({HF_CONFIG}, train) ...")
    dataset = load_dataset(
        HF_DATASET_ID,
        HF_CONFIG,
        split="train",
        cache_dir=str(cache_dir),
    )
    print(f"Loaded {len(dataset):,} ShareGPT4V records; columns: {dataset.column_names}")

    if "image" not in dataset.column_names:
        raise RuntimeError(
            "Expected an 'image' column in ShareGPT4V, found: "
            + ", ".join(dataset.column_names)
        )
    coco_dataset = dataset.filter(
        lambda row: isinstance(row["image"], str)
        and row["image"].replace("\\", "/").startswith("coco/")
    )
    print(f"ShareGPT4V COCO records: {len(coco_dataset):,}")

    if sample_count < 1:
        raise ValueError("--samples must be a positive integer")
    if sample_count > len(coco_dataset):
        raise ValueError(
            f"Requested {sample_count} samples but found only {len(coco_dataset)} COCO records."
        )

    selected_indices = random.Random(seed).sample(range(len(coco_dataset)), sample_count)
    selected = coco_dataset.select(selected_indices)
    records: list[dict[str, Any]] = []
    missing_images: list[str] = []

    coco_images_dir = data_root / "raw" / "coco" / "train2017"
    for row in selected:
        relative_image_path = row["image"].replace("\\", "/")
        record = dict(row)
        record["image"] = relative_image_path
        record["image_path"] = str(data_root / "raw" / relative_image_path)
        records.append(record)
        if not (coco_images_dir / Path(relative_image_path).name).is_file():
            missing_images.append(relative_image_path)

    manifest_path = data_root / "processed" / "sharegpt4v_coco_calibration.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "dataset": HF_DATASET_ID,
                "config": HF_CONFIG,
                "split": "train",
                "seed": seed,
                "sample_count": sample_count,
                "records": records,
            },
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    print(f"Saved {len(records)} calibration records to: {manifest_path}")
    if missing_images:
        print(
            f"Warning: {len(missing_images)} sampled image(s) are missing from "
            f"{coco_images_dir}. Download/extract COCO train2017 before using the manifest."
        )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COCO train2017 and prepare ShareGPT4V COCO samples."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Data directory (default: <repository>/data).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of reproducible calibration records to save (default: 128).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random sampling seed (default: 42)."
    )
    parser.add_argument(
        "--skip-coco-download",
        action="store_true",
        help="Skip downloading/extracting COCO images; prepare the manifest only.",
    )
    parser.add_argument(
        "--skip-sharegpt4v",
        action="store_true",
        help="Download/extract COCO images only; do not load ShareGPT4V.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_dir.expanduser().resolve()
    try:
        if not args.skip_coco_download:
            ensure_coco_images(data_root)
        if not args.skip_sharegpt4v:
            prepare_calibration_manifest(
                data_root, sample_count=args.samples, seed=args.seed
            )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
