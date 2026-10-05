"""Example: embed agentic_core in an Ollama chat app."""

import sys
from agentic_core import AgenticSession, AgenticInterrupted, build_system_prompt
from agentic_core.tools import Tool, register
from agentic_core.skills import Skill, register as register_skill


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

    # Use any Ollama-compatible client
    from ollama import Client
    client = Client(host="http://localhost:11434")

    session = AgenticSession(
        client=client,
        model="glm-5.3-flash:cloud",
        workdir=".",
        tools=True,
        skills=True,
        name="myapp",
        auto_approve_safe=False,  # ask before WRITE/EDIT/RUN
        # confirm_callback=None → uses built-in default with y/N/auto/all/always/d
        max_feedback_rounds=3,
    )

    print("Agentic session ready. Type 'quit' to exit.\n")
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
