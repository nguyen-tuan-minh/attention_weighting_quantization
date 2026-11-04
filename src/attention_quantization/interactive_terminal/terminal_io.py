"""Input helpers shared by interactive terminal screens."""

from __future__ import annotations

import select
import sys


class InteractiveTimeoutError(TimeoutError):
    """Raised when an interactive prompt remains idle past its configured limit."""


class InteractiveSessionExit(Exception):
    """Raised to stop the full investigation after the user quits the terminal UI."""


def read_timed_input(prompt: str, timeout_seconds: float) -> str:
    """Read from terminal stdin or the notebook input frontend.

    Notebook frontends such as Colab do not expose stdin as a selectable file
    descriptor. Use ``input()`` there so the frontend can display its input
    prompt; the timeout is available only when reading from a real terminal.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if not sys.stdin.isatty():
        try:
            return input().rstrip("\r\n")
        except EOFError:
            raise
    readable, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not readable:
        raise InteractiveTimeoutError
    value = sys.stdin.readline()
    if value == "":
        raise EOFError
    return value.rstrip("\r\n")
