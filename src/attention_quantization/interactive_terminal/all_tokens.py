"""Rendering helpers for the all-token terminal screen."""

from __future__ import annotations

from typing import Any

import torch

from .terminal_io import read_timed_input


def render_all_tokens(
    input_adapter: Any,
    initial_question: str,
    token_ids: list[int],
    position_distributions: list[torch.Tensor],
) -> None:
    """Print the current prompt and generated tokens with their probabilities."""
    print(f"\nInitial question: {initial_question}")
    print("\nGenerated assistant tokens:")
    for position, token_id in enumerate(token_ids, start=1):
        token_text = input_adapter.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        chosen_probability = float(position_distributions[position - 1][token_id])
        print(f"[{position}] {token_text!r}  p={chosen_probability:.4f}")


def read_all_tokens_command(token_count: int, timeout_seconds: float) -> list[str] | None:
    """Read one all-token screen command, returning None on terminal EOF."""
    try:
        command = read_timed_input(
            f"Position n [k count | p mass], r [s], e, i, h, q "
            f"(n=1-{token_count}): ",
            timeout_seconds,
        ).strip()
    except EOFError:
        return None
    return command.split()


def show_all_tokens_help() -> None:
    """Print the all-token screen command summary."""
    print(
        "Select n, n k count, or n p mass; r regenerates, r s samples, "
        "e edits the question, i shows the image, q exits."
    )
