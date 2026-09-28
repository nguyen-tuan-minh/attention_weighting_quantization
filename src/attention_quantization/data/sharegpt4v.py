"""Load ShareGPT4V records with 🤗 Datasets for LLaVA calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, Image, load_dataset


DEFAULT_ANNOTATION_RELATIVE_PATH = Path(
    "raw/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json"
)
DEFAULT_IMAGE_ROOT_RELATIVE_PATH = Path("raw")
DEFAULT_LLAVA_PROCESSOR = "llava-hf/llava-1.5-7b-hf"


def _conversation_to_fields(row: dict[str, Any], index: int) -> tuple[str, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Record {index} has no conversations list")

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
            return user_prompt, content

    raise ValueError(f"Record {index} must include a user prompt and assistant response")


def _resolve_image(image_reference: Any, image_root: Path, index: int) -> str:
    if not isinstance(image_reference, str) or not image_reference.strip():
        raise ValueError(f"Record {index} must have one image path")
    relative_path = Path(image_reference.replace("\\", "/"))
    if relative_path.is_absolute():
        raise ValueError(f"Record {index} has an absolute image path: {image_reference}")

    root = image_root.resolve()
    image_path = (root / relative_path).resolve()
    if image_path != root and root not in image_path.parents:
        raise ValueError(f"Record {index} image path escapes the image root: {image_reference}")
    return str(image_path)


def load_sharegpt4v_records(
    annotation_path: str | Path,
    image_root: str | Path,
    *,
    source_prefix: str | None = "coco/",
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Load local JSON/JSONL annotations as a Hugging Face Dataset.

    The returned dataset has columns ``record_id``, ``image`` (decoded PIL
    image), ``user_prompt``, and ``assistant_caption``. Set ``source_prefix`` to
    ``None`` to retain all image sources in the annotation file.
    """
    annotation_path = Path(annotation_path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"ShareGPT4V annotations not found: {annotation_path}")

    dataset = load_dataset(
        "json",
        data_files={"train": str(annotation_path)},
        split="train",
        cache_dir=str(Path(cache_dir).expanduser().resolve()) if cache_dir else None,
    )
    if "image" not in dataset.column_names:
        raise ValueError(f"No 'image' column in ShareGPT4V file: {annotation_path}")

    if source_prefix is not None:
        prefix = source_prefix.replace("\\", "/")
        dataset = dataset.filter(
            lambda row: isinstance(row["image"], str)
            and row["image"].replace("\\", "/").startswith(prefix),
            desc=f"Filtering ShareGPT4V records to {prefix}",
        )

    if not len(dataset):
        raise ValueError(
            f"No ShareGPT4V records matched source prefix {source_prefix!r} in {annotation_path}"
        )

    source_columns = dataset.column_names

    def normalize_record(row: dict[str, Any], index: int) -> dict[str, str]:
        user_prompt, assistant_caption = _conversation_to_fields(row, index)
        return {
            "record_id": str(row.get("id") or row.get("image") or index),
            "image": _resolve_image(row.get("image"), image_root, index),
            "user_prompt": user_prompt,
            "assistant_caption": assistant_caption,
        }

    dataset = dataset.map(
        normalize_record,
        with_indices=True,
        remove_columns=source_columns,
        desc="Normalizing ShareGPT4V conversations and image paths",
    )

    missing_paths = [path for path in dataset["image"] if not Path(path).is_file()]
    if missing_paths:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:10])
        more = f"\n  ... and {len(missing_paths) - 10} more" if len(missing_paths) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing_paths)} referenced image(s) were not found:\n{preview}{more}"
        )

    return dataset.cast_column("image", Image(decode=True))


def make_calibration_dataset(
    data_dir: str | Path,
    *,
    sample_count: int = 128,
    seed: int = 42,
    annotation_path: str | Path | None = None,
    image_root: str | Path | None = None,
    source_prefix: str | None = "coco/",
) -> Dataset:
    """Return a deterministic, no-replacement calibration subset.

    Images are decoded lazily by the Datasets ``Image`` feature. Each selected
    row retains the ShareGPT4V user prompt and assistant caption.
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
    cache_dir = data_dir / "cache" / "datasets"

    dataset = load_sharegpt4v_records(
        annotation_path,
        image_root,
        source_prefix=source_prefix,
        cache_dir=cache_dir,
    )
    if sample_count > len(dataset):
        raise ValueError(
            f"Requested {sample_count} calibration records, but only {len(dataset)} "
            "matching records are available."
        )
    return dataset.shuffle(seed=seed).select(range(sample_count))


def load_llava_processor(
    model_name_or_path: str = DEFAULT_LLAVA_PROCESSOR,
    *,
    do_pad: bool = True,
):
    """Load the Transformers processor matching the LLaVA 1.5 checkpoint."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "do_pad"):
        image_processor.do_pad = do_pad
    return processor


class LlavaCalibrationCollator:
    """Use the Transformers LLaVA processor and mark assistant tokens as targets.

    Prompt formatting follows the LLaVA 1.5 ``USER: <image> ... ASSISTANT:``
    format. Padding and image preprocessing are handled by the processor.
    """

    def __init__(self, processor: Any, *, ignore_index: int = -100) -> None:
        self.processor = processor
        self.ignore_index = ignore_index

    @staticmethod
    def _without_image_marker(user_prompt: str) -> str:
        # Add exactly one image marker in the LLaVA prompt, even if the source
        # conversation contains a marker on a separate line.
        return user_prompt.replace("<image>", "").strip()

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty sample list")

        images = []
        full_prompts = []
        prefix_prompts = []
        for sample in samples:
            sample_images = sample["image"]
            images.append(sample_images)
            user_text = self._without_image_marker(sample["user_prompt"])
            prefix = f"USER: <image>\n{user_text} ASSISTANT:"
            prefix_prompts.append(prefix)
            full_prompts.append(f"{prefix} {sample['assistant_caption']}</s>")

        batch = self.processor(
            text=full_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        # Determine each assistant boundary with the same processor and image,
        # so image-token expansion is included in the prefix length.
        for row_index, (prefix, image) in enumerate(zip(prefix_prompts, images)):
            prefix_batch = self.processor(
                text=prefix,
                images=image,
                padding=False,
                return_tensors="pt",
            )
            prefix_length = prefix_batch["input_ids"].shape[-1]
            labels[row_index, :prefix_length] = self.ignore_index

        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = self.ignore_index
        batch["labels"] = labels
        batch["record_ids"] = [sample["record_id"] for sample in samples]
        return batch
