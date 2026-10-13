"""Rendering helpers for the candidate inspection terminal screen."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from .terminal_io import read_timed_input


def render_candidates(
    input_adapter: Any,
    position: int,
    position_options: list[tuple[int, float]],
) -> None:
    """Print the ranked token candidates for one generated position."""
    print(f"\nTop candidates at position [{position}]:")
    for rank, (candidate_id, probability) in enumerate(position_options, start=1):
        candidate_text = input_adapter.tokenizer.decode(
            [candidate_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(f"[{rank}] {candidate_text!r}  p={probability:.4f}")


def read_candidate_command(candidate_count: int, timeout_seconds: float) -> list[str] | None:
    """Read one candidate-screen command, returning None on terminal EOF."""
    try:
        command = read_timed_input(
            f"Candidate: n [i|t [c]|g], n + c, or n replace [c] (n=1-{candidate_count}), "
            "h for help, b to list positions, q to finish: ",
            timeout_seconds,
        ).strip().lower().split()
    except EOFError:
        return None
    return command


def show_candidate_help() -> None:
    """Print the candidate-screen command summary."""
    print(
        "Use n or n i for image scores, n t for all prior text scores, "
        "n t c for the latest c, n g for image/text gradient percentages, "
        "n + c to preview a continuation, "
        "n replace [c] to replace and regenerate, b to return, q to exit."
    )


def show_gradient_norm_percentages(
    image_scores: dict[int, Any],
    text_scores: dict[int, Any],
    title: str,
) -> None:
    """Plot per-layer image/text shares after root-summing each modality's squared scores."""
    layers = sorted(set(image_scores) | set(text_scores))
    image_norms = [float(image_scores[layer].sum().sqrt()) if layer in image_scores else 0.0 for layer in layers]
    text_norms = [float(text_scores[layer].sum().sqrt()) if layer in text_scores else 0.0 for layer in layers]
    totals = [image_value + text_value for image_value, text_value in zip(image_norms, text_norms)]
    image_percentages = [100.0 * value / total if total else 0.0 for value, total in zip(image_norms, totals)]
    text_percentages = [100.0 * value / total if total else 0.0 for value, total in zip(text_norms, totals)]

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.bar(layers, image_percentages, label="Image", color="tab:blue")
    axis.bar(layers, text_percentages, bottom=image_percentages, label="Text", color="tab:orange")
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Share of image + text gradient norm (%)")
    axis.set_ylim(0, 100)
    axis.set_xticks(layers)
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    plt.show()
    plt.close(figure)
