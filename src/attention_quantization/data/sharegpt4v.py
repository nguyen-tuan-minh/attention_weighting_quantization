"""Load ShareGPT4V and download its related image datasets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, Image, load_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "dataset.yaml"
SHAREGPT4V_DATASET_ID = "Lin-Chen/ShareGPT4V"
SHAREGPT4V_CONFIG = "ShareGPT4V"

# Some ShareGPT4V sources require a separate or manual download.
IMAGE_DATASETS: dict[str, tuple[tuple[str, str], ...]] = {
    "coco": (("http://images.cocodataset.org/zips/train2017.zip", "coco/train2017.zip"),),
    "gqa": (("https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip", "gqa/images.zip"),),
    "textvqa": (
        ("https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip", "textvqa/train_val_images.zip"),
    ),
    "visual-genome": (
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "vg/images.zip"),
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "vg/images2.zip"),
    ),
}
SOURCE_PREFIXES = {
    "coco": "coco/",
    "gqa": "gqa/",
    "textvqa": "textvqa/",
    "visual-genome": "vg/",
}
COCO_TRAIN2017_MIN_FILES = 100_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_data_paths(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Path]:
    """Read dataset paths from YAML, resolving relative paths from the repository root."""
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file) or {}

    paths = {}
    for key in ("data_root", "raw_dir", "cache_dir", "processed_dir"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {key!r} in dataset config: {config_path}")
        path = Path(value).expanduser()
        paths[key] = (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()
    return paths


def _get_image_download_settings(config_path: str | Path) -> tuple[int | None, int | None]:
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file) or {}

    max_images = config.get("max_images")
    seed = config.get("download_seed")
    if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
        raise ValueError("max_images in dataset config must be a positive integer or null")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("download_seed in dataset config must be an integer or null")
    return max_images, seed


def load_sharegpt4v_dataset(
    *,
    split: str = "train",
    source: str | None = "coco",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cache_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    download_images: bool = True,
    max_images: int | None = None,
    seed: int | None = None,
    existing_images_only: bool = False,
) -> Dataset:
    """Load ShareGPT4V with local images represented by a lazy ``Image`` feature.

    COCO is selected and downloaded by default. Images remain on disk and are
    decoded on access, so this does not load the full image collection into RAM.
    Set ``source=None`` to keep every record, or select another supported image
    source. Set ``download_images=False`` to use files already present locally.
    ``max_images`` and ``seed`` default to the values in dataset config.
    Set ``existing_images_only=True`` to omit records whose local image is missing.
    """
    if source is not None and source not in SOURCE_PREFIXES:
        raise ValueError(f"Unsupported source {source!r}; choose from {tuple(SOURCE_PREFIXES)} or None")

    paths = get_data_paths(config_path)
    resolved_cache = Path(cache_dir).expanduser().resolve() if cache_dir else paths["cache_dir"]
    resolved_image_dir = Path(image_dir).expanduser().resolve() if image_dir else paths["raw_dir"]

    if download_images and source is not None:
        download_image_dataset(
            source,
            config_path=config_path,
            raw_dir=resolved_image_dir,
            max_images=max_images,
            seed=seed,
        )

    dataset = load_dataset(
        SHAREGPT4V_DATASET_ID,
        SHAREGPT4V_CONFIG,
        split=split,
        cache_dir=str(resolved_cache),
    )
    if "image" not in dataset.column_names:
        raise ValueError("ShareGPT4V dataset does not contain an 'image' column")

    if source is not None:
        prefix = SOURCE_PREFIXES[source]
        dataset = dataset.filter(
            lambda row: isinstance(row["image"], str)
            and row["image"].replace("\\", "/").startswith(prefix),
            desc=f"Filtering ShareGPT4V to {source}",
        )

    def resolve_image_path(row: dict[str, Any]) -> dict[str, str]:
        reference = row["image"].replace("\\", "/")
        relative_path = Path(reference)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe ShareGPT4V image path: {reference}")
        return {"image": str((resolved_image_dir / relative_path).resolve())}

    dataset = dataset.map(resolve_image_path, desc="Resolving local ShareGPT4V image paths")
    if existing_images_only:
        dataset = dataset.cast_column("image", Image(decode=False))
        total_records = len(dataset)

        def image_exists(row: dict[str, Any]) -> bool:
            image_value = row.get("image")
            image_path = image_value.get("path") if isinstance(image_value, dict) else None
            return isinstance(image_path, str) and Path(image_path).is_file()

        dataset = dataset.filter(image_exists, desc="Filtering to records with local images")
        print(f"Records with local images: {len(dataset):,}/{total_records:,}")
    return dataset.cast_column("image", Image(decode=True))


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)

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


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _extract_zip(
    archive_path: Path,
    destination: Path,
    max_images: int | None = None,
    seed: int | None = None,
) -> int:
    """Extract a zip safely, optionally limiting newly extracted image files."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    extracted_images = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        if max_images is None:
            archive.extractall(destination)
            return sum(
                1
                for member in members
                if not member.is_dir() and Path(member.filename).suffix.lower() in IMAGE_SUFFIXES
            )

        candidates = [
            member
            for member in members
            if not member.is_dir()
            and Path(member.filename).suffix.lower() in IMAGE_SUFFIXES
            and not (destination / member.filename).is_file()
        ]
        if len(candidates) > max_images:
            candidates = random.Random(seed).sample(candidates, max_images)
        chosen_images = {member.filename for member in candidates}

        for member in members:
            if member.is_dir():
                continue
            is_image = Path(member.filename).suffix.lower() in IMAGE_SUFFIXES
            if is_image and member.filename not in chosen_images:
                continue
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if is_image:
                extracted_images += 1
    return extracted_images


def download_image_dataset(
    dataset_name: str = "coco",
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    raw_dir: str | Path | None = None,
    max_images: int | None = None,
    seed: int | None = None,
) -> Path:
    """Download and extract one image source under the configured raw data directory."""
    if dataset_name not in IMAGE_DATASETS:
        raise ValueError(f"Unsupported image dataset {dataset_name!r}; choose from {tuple(IMAGE_DATASETS)}")
    configured_max_images, configured_seed = _get_image_download_settings(config_path)
    max_images = configured_max_images if max_images is None else max_images
    seed = configured_seed if seed is None else seed
    if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
        raise ValueError("max_images must be a positive integer or None")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer or None")

    resolved_raw_dir = (
        Path(raw_dir).expanduser().resolve()
        if raw_dir is not None
        else get_data_paths(config_path)["raw_dir"]
    )
    for url, relative_archive_path in IMAGE_DATASETS[dataset_name]:
        archive_path = resolved_raw_dir / relative_archive_path
        extract_dir = archive_path.parent
        extraction_marker = extract_dir / f".{archive_path.name}.extracted"
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
        partial_path.unlink(missing_ok=True)
        existing_images = _count_images(extract_dir)
        if max_images is not None and existing_images >= max_images:
            print(f"Already extracted: {archive_path.name}")
            extraction_marker.write_text(
                json.dumps({"max_images": max_images, "image_count": existing_images}) + "\n",
                encoding="utf-8",
            )
            if archive_path.exists():
                archive_path.unlink()
                print(f"Removed redundant archive: {archive_path}")
            continue

        marker_data: dict[str, Any] = {}
        if extraction_marker.exists():
            try:
                saved_marker = json.loads(extraction_marker.read_text(encoding="utf-8"))
                marker_data = saved_marker if isinstance(saved_marker, dict) else {}
            except (OSError, json.JSONDecodeError):
                marker_data = {}
        marker_means_full_extraction = marker_data.get("max_images") is None and (
            "max_images" in marker_data or existing_images >= COCO_TRAIN2017_MIN_FILES
        )
        legacy_full_extraction = (
            not marker_data
            and extraction_marker.exists()
            and existing_images >= COCO_TRAIN2017_MIN_FILES
        )
        if max_images is None and (marker_means_full_extraction or legacy_full_extraction):
            print(f"Already extracted: {archive_path.name}")
            if archive_path.exists():
                archive_path.unlink()
                print(f"Removed redundant archive: {archive_path}")
            continue

        if archive_path.exists():
            print(f"Using existing archive: {archive_path}")
        else:
            print(f"Downloading {dataset_name} archive to: {archive_path}")
            _download_file(url, archive_path)
        print(f"Extracting {archive_path.name} into: {extract_dir}")
        remaining_images = None if max_images is None else max_images - existing_images
        extracted_images = _extract_zip(archive_path, extract_dir, remaining_images, seed)
        extraction_marker.write_text(
            json.dumps(
                {
                    "max_images": max_images,
                    "seed": seed,
                    "image_count": _count_images(extract_dir),
                    "newly_extracted": extracted_images,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        archive_path.unlink()
        print(f"Removed extracted archive: {archive_path}")
    print(f"{dataset_name} images are ready under: {resolved_raw_dir}")
    return resolved_raw_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download image files used by ShareGPT4V.")
    parser.add_argument(
        "--dataset",
        choices=tuple(IMAGE_DATASETS),
        default="coco",
        help="Image dataset to download (default: coco).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML dataset path config (default: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Override configured maximum image count (dataset.yaml default: 1024; null in config means all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override download_seed in dataset.yaml (default: null, meaning choose a new random subset).",
    )
    args = parser.parse_args()
    try:
        download_image_dataset(
            args.dataset,
            config_path=args.config,
            max_images=args.max_images,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
