"""Small configuration and path helpers shared by project scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, raising a useful error for other YAML values."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return value


def repository_path(value: str | Path) -> Path:
    """Resolve an absolute path as-is or a relative path from the repository."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()
