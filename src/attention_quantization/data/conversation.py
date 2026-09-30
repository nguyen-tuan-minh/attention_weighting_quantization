"""Helpers for reading the common user/assistant conversation format."""

from __future__ import annotations

from typing import Any


def get_user_assistant_text(sample: dict[str, Any]) -> tuple[str, str]:
    """Return the first user prompt and its following assistant response."""
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Sample has no 'conversations' list")

    user_text = None
    assistant_text = None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        text = message.get("value", message.get("content"))
        if not isinstance(text, str):
            continue
        if role in {"human", "user"} and user_text is None:
            user_text = text.replace("<image>", "").strip()
        elif role in {"gpt", "assistant"} and user_text is not None:
            assistant_text = text.strip()
            break

    if not user_text or assistant_text is None:
        raise ValueError("Sample needs a user prompt and assistant response")
    return user_text, assistant_text
