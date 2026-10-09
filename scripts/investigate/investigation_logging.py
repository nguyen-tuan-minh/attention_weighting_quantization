"""Shared verbosity and warning controls for investigation scripts."""

from __future__ import annotations

import warnings
from typing import Any

# CLI level schema: accepted log-level string -> increasing verbosity threshold.
_LEVELS = {"none": 0, "normal": 1, "extensive": 2}
_CURRENT_LEVEL = _LEVELS["normal"]


def configure_logging(level: str, *, quiet_warnings: bool = False) -> None:
    """Set the script log threshold and optional dependency warning filters."""
    global _CURRENT_LEVEL
    _CURRENT_LEVEL = _LEVELS[level]
    if quiet_warnings:
        warnings.filterwarnings("ignore")
    verbosity_name = "error" if quiet_warnings or level == "none" else (
        "info" if level == "extensive" else "warning"
    )
    try:
        from transformers.utils import logging as transformers_logging

        getattr(transformers_logging, f"set_verbosity_{verbosity_name}")()
    except ImportError:
        pass
    try:
        from datasets.utils import logging as datasets_logging

        getattr(datasets_logging, f"set_verbosity_{verbosity_name}")()
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import logging as hub_logging

        getattr(hub_logging, f"set_verbosity_{verbosity_name}")()
    except ImportError:
        pass


def log(message: Any, *, level: str = "normal", **kwargs: Any) -> None:
    """Print a message when its verbosity is enabled."""
    if _CURRENT_LEVEL >= _LEVELS[level]:
        print(message, **kwargs)
