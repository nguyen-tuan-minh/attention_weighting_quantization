"""Input helpers shared by interactive terminal screens."""

from __future__ import annotations

import select
import sys


class InteractiveTimeoutError(TimeoutError):
    """Raised when an interactive prompt remains idle past its configured limit."""


class InteractiveSessionExit(Exception):
    """Raised to stop the full investigation after the user quits the terminal UI."""


def read_timed_input(prompt: str, timeout_seconds: float) -> str:
    """Read one line from stdin, raising TimeoutError when the prompt expires."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    readable, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not readable:
        raise InteractiveTimeoutError
    value = sys.stdin.readline()
    if value == "":
        raise EOFError
    return value.rstrip("\r\n")
