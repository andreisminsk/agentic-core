"""Shared utilities for action execution.

Extracted from actions.py — print helpers and text truncation.
"""

import sys

from .constants import ANSI_AGENT, ANSI_RESET, ANSI_TOOL


def agent_print(*args, **kwargs):
    """Print with agent color when stdout is a TTY."""
    if sys.stdout.isatty():
        print(ANSI_AGENT, end="", flush=True)
        print(*args, **kwargs)
        print(ANSI_RESET, end="", flush=True)
    else:
        print(*args, **kwargs)


def tool_print(*args, **kwargs):
    """Print with tool color when stdout is a TTY."""
    if sys.stdout.isatty():
        print(ANSI_TOOL, end="", flush=True)
        print(*args, **kwargs)
        print(ANSI_RESET, end="", flush=True)
    else:
        print(*args, **kwargs)


def _truncate(text, max_chars):
    """Truncate text to max_chars, adding a notice if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[... truncated, {len(text)} chars total]"
