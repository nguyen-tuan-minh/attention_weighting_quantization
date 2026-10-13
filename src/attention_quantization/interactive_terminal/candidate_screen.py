"""Rendering helpers for the candidate inspection terminal screen."""

from __future__ import annotations

from typing import Any

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
            f"Candidate: n [i|t [c]], n + c, or n replace [c] (n=1-{candidate_count}), "
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
        "n t c for the latest c, n + c to preview a continuation, "
        "n replace [c] to replace and regenerate, b to return, q to exit."
    )
