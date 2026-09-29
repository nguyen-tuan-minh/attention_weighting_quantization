"""Load ShareGPT4V and download its related image datasets."""

from __future__ import annotations

import argparse
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


def load_sharegpt4v_dataset(
    *,
    split: str = "train",
    source: str | None = "coco",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cache_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    download_images: bool = True,
) -> Dataset:
    """Load ShareGPT4V with local images represented by a lazy ``Image`` feature.

    COCO is selected and downloaded by default. Images remain on disk and are
    decoded on access, so this does not load the full image collection into RAM.
    Set ``source=None`` to keep every record, or select another supported image
    source. Set ``download_images=False`` to use files already present locally.
    """
    if source is not None and source not in SOURCE_PREFIXES:
        raise ValueError(f"Unsupported source {source!r}; choose from {tuple(SOURCE_PREFIXES)} or None")

    paths = get_data_paths(config_path)
    resolved_cache = Path(cache_dir).expanduser().resolve() if cache_dir else paths["cache_dir"]
    resolved_image_dir = Path(image_dir).expanduser().resolve() if image_dir else paths["raw_dir"]

    if download_images and source is not None:
        download_image_dataset(source, config_path=config_path, raw_dir=resolved_image_dir)

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
    return dataset.cast_column("image", Image(decode=True))


def _download_file(url: str, destination: Path) -> None:
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


def _extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract zip contents while rejecting paths that escape the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def download_image_dataset(
    dataset_name: str = "coco",
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    raw_dir: str | Path | None = None,
) -> Path:
    """Download and extract one image source under the configured raw data directory."""
    if dataset_name not in IMAGE_DATASETS:
        raise ValueError(f"Unsupported image dataset {dataset_name!r}; choose from {tuple(IMAGE_DATASETS)}")

    resolved_raw_dir = (
        Path(raw_dir).expanduser().resolve()
        if raw_dir is not None
        else get_data_paths(config_path)["raw_dir"]
    )
    for url, relative_archive_path in IMAGE_DATASETS[dataset_name]:
        archive_path = resolved_raw_dir / relative_archive_path
        extract_dir = archive_path.parent
        extraction_marker = extract_dir / f".{archive_path.name}.extracted"
        if archive_path.exists():
            print(f"Using existing archive: {archive_path}")
        else:
            print(f"Downloading {dataset_name} archive to: {archive_path}")
            _download_file(url, archive_path)
        if extraction_marker.exists():
            print(f"Already extracted: {archive_path.name}")
        else:
            print(f"Extracting {archive_path.name} into: {extract_dir}")
            _extract_zip(archive_path, extract_dir)
            extraction_marker.touch()
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
    args = parser.parse_args()
    try:
        download_image_dataset(args.dataset, config_path=args.config)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
