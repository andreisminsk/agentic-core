"""Example: embed agentic_core with an OpenAI-compatible API + TPM limiting.

Demonstrates using OpenAIBackend with tpm_limit to proactively throttle
requests on rate-limited endpoints. When tpm_limit is set, the backend
auto-configures a coordinated bundle of middleware:

  - HistoryTrimmer: caps context to max_context (default 16384) — smaller
    context means fewer tokens per request, stretching the TPM budget.
  - TPMTracker: rolling 60s window, waits before sending if over budget.
  - RetryHandler: aggressive — retries on 429/rate_limit/connection with
    20s jittered backoff, respects Retry-After headers.

Without tpm_limit, trimming uses full ctx_size and retry is
connection-errors-only (see example-openai.py).

Requires:
    pip install openai

Usage:
    Set OPENAI_API_KEY and adjust the variables below.
    python example-openai-tpm.py
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

    # Configure the OpenAI-compatible backend with TPM limiting
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ["UHU_PROJECTS_KEY"]
    model = os.environ.get("OPENAI_MODEL", "gpt-5.1")

    # TPM limit — the master switch that activates the full middleware bundle.
    # Set to your endpoint's tokens-per-minute budget (e.g. OpenAI tier 1: 30K-500K).
    tpm_limit = int(os.environ.get("OPENAI_TPM_LIMIT", "50000"))

    # ctx_size: the model's full context window (e.g. gpt-5.1 supports 128K).
    # This is the upper bound — the trimmer will never let history exceed this.
    # It's also used for the max_completion_tokens API parameter (capped at 8192).
    ctx_size = 131072  # 128K — the model's actual context window size

    # max_context: the trim budget cap used when tpm_limit is set.
    # When TPM is active, the trimmer uses min(ctx_size, max_context) instead
    # of the full ctx_size. A smaller max_context means fewer tokens per
    # request, which stretches the TPM budget further — but less context
    # means the model sees less conversation history.
    # Without tpm_limit, max_context is ignored and the trimmer uses full ctx_size.
    max_context = 16384  # 16K — cap history at 16K tokens to conserve TPM budget

    backend = OpenAIBackend(
        base_url=base_url,
        api_key=api_key,
        model=model,
        ctx_size=ctx_size,        # model's full context window (upper bound for trimming)
        temperature=0.0,
        tpm_limit=tpm_limit,      # activates: capped trimming + TPM tracker + aggressive retry
        max_context=max_context,  # trim history to 16K tokens (smaller = fewer tokens/request)
        quiet=False,               # show TPM wait messages
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
    print(f"TPM limit: {tpm_limit} tokens/min | Context window: {ctx_size} | Trim cap: {max_context} tokens")
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
