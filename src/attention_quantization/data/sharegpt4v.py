"""Read ShareGPT4V annotations and expose reproducible calibration samples.

The dataset returns the image(s), human prompt, and assistant caption separately.
This keeps the data layer independent of LLaVA prompt templating and tokenization;
the model adapter can form the teacher-forced calibration input from all three.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ANNOTATION_RELATIVE_PATH = Path(
    "raw/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json"
)
DEFAULT_IMAGE_ROOT_RELATIVE_PATH = Path("raw")


@dataclass(frozen=True)
class ShareGPT4VRecord:
    """One image/conversation pair from a ShareGPT4V annotation file."""

    record_id: str
    image_paths: tuple[Path, ...]
    user_prompt: str
    assistant_caption: str


def _read_annotation_rows(annotation_path: Path) -> list[dict[str, Any]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"ShareGPT4V annotations not found: {annotation_path}")

    if annotation_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with annotation_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {annotation_path}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object on line {line_number} of {annotation_path}")
                rows.append(row)
        return rows

    if annotation_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json or .jsonl annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data", payload.get("records"))
    else:
        rows = None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(
            "Expected the annotation JSON to contain a list of records, either at the "
            "top level or under a 'data'/'records' key."
        )
    return rows


def _conversation_texts(row: dict[str, Any], row_number: int) -> tuple[str, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Record {row_number} has no conversations list")

    user_prompt: str | None = None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        content = message.get("value", message.get("content"))
        if not isinstance(content, str):
            continue
        if role in {"human", "user"} and user_prompt is None:
            user_prompt = content
        elif role in {"gpt", "assistant"} and user_prompt is not None:
            # The first assistant turn is the caption target. Keep the human text
            # unchanged, including any <image> marker expected by LLaVA formatting.
            return user_prompt, content

    raise ValueError(f"Record {row_number} must include a user prompt and assistant response")


def _image_references(row: dict[str, Any], row_number: int) -> tuple[str, ...]:
    image_value = row.get("image")
    if isinstance(image_value, str):
        references = (image_value,)
    elif isinstance(image_value, list) and image_value and all(
        isinstance(path, str) for path in image_value
    ):
        references = tuple(image_value)
    else:
        raise ValueError(f"Record {row_number} must have an image path or list of image paths")
    if not all(reference.strip() for reference in references):
        raise ValueError(f"Record {row_number} contains an empty image path")
    return references


def _resolve_image_path(reference: str, image_root: Path, row_number: int) -> Path:
    relative_path = Path(reference.replace("\\", "/"))
    if relative_path.is_absolute():
        raise ValueError(f"Record {row_number} has an absolute image path: {reference}")
    resolved = (image_root / relative_path).resolve()
    root = image_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Record {row_number} image path escapes the image root: {reference}")
    return resolved


def load_sharegpt4v_records(
    annotation_path: str | Path,
    image_root: str | Path,
    *,
    source_prefix: str | None = "coco/",
    check_images: bool = True,
) -> list[ShareGPT4VRecord]:
    """Load records, optionally filter by image-path prefix, and validate paths.

    ``image_root`` is the directory above the source path stored in the JSON.
    For COCO annotations such as ``coco/train2017/000000000001.jpg``, use the
    repository data directory's ``data/raw`` folder as the image root.
    """
    annotation_path = Path(annotation_path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    rows = _read_annotation_rows(annotation_path)
    records: list[ShareGPT4VRecord] = []
    missing_images: list[str] = []

    for row_number, row in enumerate(rows):
        try:
            references = _image_references(row, row_number)
        except ValueError:
            # Other rows can belong to text-only data or have a different schema.
            # For the selected source, malformed records should be reported.
            if source_prefix is None:
                raise
            continue

        normalized_references = tuple(reference.replace("\\", "/") for reference in references)
        if source_prefix is not None and not any(
            reference.startswith(source_prefix) for reference in normalized_references
        ):
            continue

        try:
            user_prompt, assistant_caption = _conversation_texts(row, row_number)
            resolved_paths = tuple(
                _resolve_image_path(reference, image_root, row_number)
                for reference in references
            )
        except ValueError as error:
            raise ValueError(f"Invalid ShareGPT4V record at row {row_number}: {error}") from error

        if check_images:
            missing_images.extend(str(path) for path in resolved_paths if not path.is_file())

        record_id = str(row.get("id", row.get("image", row_number)))
        records.append(
            ShareGPT4VRecord(
                record_id=record_id,
                image_paths=resolved_paths,
                user_prompt=user_prompt,
                assistant_caption=assistant_caption,
            )
        )

    if missing_images:
        preview = "\n".join(f"  - {path}" for path in missing_images[:10])
        more = f"\n  ... and {len(missing_images) - 10} more" if len(missing_images) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing_images)} referenced image(s) were not found under {image_root}:\n"
            f"{preview}{more}"
        )
    if not records:
        filter_description = repr(source_prefix) if source_prefix is not None else "no source filter"
        raise ValueError(
            f"No usable ShareGPT4V records found in {annotation_path} for {filter_description}."
        )
    return records


class ShareGPT4VCalibrationDataset:
    """Indexable dataset with lazy image loading for model-specific collators.

    Each item contains ``record_id``, ``images`` (RGB PIL images),
    ``user_prompt``, and ``assistant_caption``. The assistant caption is retained
    as the calibration target; a separate fixed prompt belongs to generation
    quality evaluation.
    """

    def __init__(self, records: list[ShareGPT4VRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Pillow is required to open images. Install it with: python -m pip install Pillow"
            ) from error

        record = self.records[index]
        images = []
        for image_path in record.image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        return {
            "record_id": record.record_id,
            "images": images,
            "user_prompt": record.user_prompt,
            "assistant_caption": record.assistant_caption,
        }


def make_calibration_dataset(
    data_dir: str | Path,
    *,
    sample_count: int = 128,
    seed: int = 42,
    annotation_path: str | Path | None = None,
    image_root: str | Path | None = None,
    source_prefix: str | None = "coco/",
) -> ShareGPT4VCalibrationDataset:
    """Create a deterministic, no-replacement sample for calibration.

    The default paths match ``scripts/download_dataset.py``. Image pixels are
    opened lazily when an item is requested, rather than retained in memory.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be a positive integer")

    data_dir = Path(data_dir).expanduser().resolve()
    annotation_path = (
        Path(annotation_path).expanduser().resolve()
        if annotation_path is not None
        else data_dir / DEFAULT_ANNOTATION_RELATIVE_PATH
    )
    image_root = (
        Path(image_root).expanduser().resolve()
        if image_root is not None
        else data_dir / DEFAULT_IMAGE_ROOT_RELATIVE_PATH
    )

    records = load_sharegpt4v_records(
        annotation_path,
        image_root,
        source_prefix=source_prefix,
    )
    if sample_count > len(records):
        raise ValueError(
            f"Requested {sample_count} calibration records, but only {len(records)} "
            "matching records are available."
        )

    selected_indices = random.Random(seed).sample(range(len(records)), sample_count)
    return ShareGPT4VCalibrationDataset([records[index] for index in selected_indices])
