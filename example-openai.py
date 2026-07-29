"""Example: embed agentic_core with an OpenAI-compatible API (no TPM limit).

Demonstrates using OpenAIBackend to connect to any OpenAI-compatible endpoint
(OpenAI, Ollama /v1, vLLM, LM Studio, etc.) without tokens-per-minute limiting.

Requires:
    pip install openai

Usage:
    Set OPENAI_API_KEY (or OLLAMA_API_KEY for local Ollama /v1).
    python example-openai.py
"""

import os
import sys

from agentic_core import AgenticSession, AgenticInterrupted
from agentic_core.tools import Tool, register
from agentic_core.skills import Skill, register as register_skill
from agentic_core.backends import OpenAIBackend


# ── Custom tool example ─────────────────────────────────────────────────
class EchoTool(Tool):
    name = "echo"
    description = "Echo back the provided text."
    system_prompt = (
        "## echo\n"
        "Echo back text. Parameters: text (string, required)."
    )

    def execute(self, params, workdir=None):
        return f"Echo: {params.get('text', '')}"


# ── Custom skill example ───────────────────────────────────────────────
class GreetingSkill(Skill):
    name = "greeting"
    description = "Generate a greeting."
    system_prompt = "When invoked, produce a friendly greeting for the user."
    parameters = {"name": {"type": "string", "required": False, "description": "Person to greet"}}

    def execute(self, params, workdir=None, session=None):
        name = params.get("name", "friend")
        return f"[Skill greeting invoked] Hello, {name}!"


def main():
    # Register custom tools/skills
    register(EchoTool())
    register_skill(GreetingSkill())

    # Configure the OpenAI-compatible backend
    # Defaults: local Ollama /v1 endpoint. Override with env vars for other providers.
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434")
    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("OLLAMA_API_KEY", "ollama"))
    model = os.environ.get("OPENAI_MODEL", "glm-5.1:cloud")

    backend = OpenAIBackend(
        base_url=base_url,
        api_key=api_key,
        model=model,
        ctx_size=32768,
        temperature=0.0,
        # tpm_limit=None → light retry (connection errors only), full-ctx trimming
    )

    session = AgenticSession(
        backend=backend,
        workdir=".",
        tools=True,
        skills=True,
        name="myapp",
        auto_approve_safe=False,  # ask before WRITE/EDIT/RUN
        max_feedback_rounds=3,
    )

    print(f"Agentic session ready (OpenAI backend: {base_url}, model: {model}).")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        try:
            response = session.run(user_input)
        except AgenticInterrupted as e:
            print(f"\n[interrupted during {e.phase}]")
            if e.partial_text:
                print(f"  partial: {e.partial_text[:200]}")
            continue
        # Streaming already displayed the response; just add a separator
        print()


if __name__ == "__main__":
    main()
