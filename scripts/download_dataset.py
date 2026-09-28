"""Download ShareGPT4V caption annotations and related image datasets.

Examples:
    python scripts/download_dataset.py                 # COCO (default)
    python scripts/download_dataset.py --dataset gqa
    python scripts/download_dataset.py --dataset textvqa
    python scripts/download_dataset.py --dataset visual-genome

The default downloads the ShareGPT4V GPT-4V caption annotations and COCO
train2017 images. Sampling, seeds, and calibration-set preparation belong to
a later data-preparation step.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHAREGPT4V_ANNOTATION = "sharegpt4v_instruct_gpt4-vision_cap100k.json"
SHAREGPT4V_ANNOTATION_URL = (
    "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/"
    + SHAREGPT4V_ANNOTATION
)

# Public image archives linked by the ShareGPT4V data documentation. Some
# ShareGPT4V sources require a separate/manual download and are not listed here.
DATASETS: dict[str, tuple[tuple[str, str], ...]] = {
    "coco": (
        ("http://images.cocodataset.org/zips/train2017.zip", "coco/train2017.zip"),
    ),
    "gqa": (
        ("https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip", "gqa/images.zip"),
    ),
    "textvqa": (
        (
            "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
            "textvqa/train_val_images.zip",
        ),
    ),
    "visual-genome": (
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "vg/images.zip"),
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "vg/images2.zip"),
    ),
}


def download_file(url: str, destination: Path) -> None:
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


def extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract a zip archive, rejecting paths that escape the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def download_dataset(dataset_name: str, data_root: Path) -> None:
    for url, relative_archive_path in DATASETS[dataset_name]:
        archive_path = data_root / "raw" / relative_archive_path
        extract_dir = archive_path.parent
        if archive_path.exists():
            print(f"Using existing archive: {archive_path}")
        else:
            print(f"Downloading {dataset_name} archive to: {archive_path}")
            download_file(url, archive_path)

        print(f"Extracting {archive_path.name} into: {extract_dir}")
        extract_zip(archive_path, extract_dir)
    print(f"{dataset_name} download is ready under: {data_root / 'raw'}")


def download_sharegpt4v_captions(data_root: Path) -> Path:
    """Download the 100K ShareGPT4V GPT-4V caption annotation file."""
    destination = data_root / "raw" / "sharegpt4v" / SHAREGPT4V_ANNOTATION
    if destination.exists():
        print(f"ShareGPT4V caption annotations already exist: {destination}")
    else:
        print(f"Downloading ShareGPT4V caption annotations to: {destination}")
        download_file(SHAREGPT4V_ANNOTATION_URL, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a ShareGPT4V-related image dataset into the repository."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="coco",
        help="Image dataset to download (default: coco). Run once per dataset.",
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Skip downloading the ShareGPT4V caption annotation JSON.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Data directory (default: <repository>/data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_dir.expanduser().resolve()
    try:
        if not args.skip_captions:
            download_sharegpt4v_captions(data_root)
        download_dataset(args.dataset, data_root)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
