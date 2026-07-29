"""Minimalistic agentic core for Ollama LLM chat apps.

Provides a block-based action protocol (WRITE/EDIT/FILE/RUN/TOOL/SKILL)
that any Ollama chat application can embed to gain agentic file/shell/tool
capabilities.

Quick start:
    from agentic_core import AgenticSession
    session = AgenticSession(client, model="glm-5.1:cloud", workdir=".")
    session.run("Create a hello.py file")
"""

from .session import AgenticSession
from .parser import parse_actions, parse_edit_content
from .system_prompt import build_system_prompt
from .exceptions import AgenticInterrupted
from .backends import (
    Backend, OllamaBackend, OpenAIBackend,
    TokenCounter, TPMTracker, RetryHandler, HistoryTrimmer,
)

__all__ = [
    "AgenticSession", "parse_actions", "parse_edit_content",
    "build_system_prompt", "AgenticInterrupted",
    "Backend", "OllamaBackend", "OpenAIBackend",
    "TokenCounter", "TPMTracker", "RetryHandler", "HistoryTrimmer",
]
__version__ = "0.1.0"
