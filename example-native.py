"""Example: native tool calling (solicited mode) with agentic-core.

Run from the agentic-core repo root:

    python example-native.py

Demonstrates the two solicited native modes (text-channel is the default and
needs no example — every other example here IS text mode):

  - "hybrid" (recommended default for tool-capable models):
      * tool schemas are attached to the LLM request (the `tools` param), so
        function-calling models emit structured tool_calls;
      * the prompt PREFERs text action blocks (FILE:/WRITE:/EDIT:) for file
        content — the transparent path (diffs, review, rollback) — with
        read_file/write_file/replace_in_file kept on the wire as BACKUP:
        if a text block fails to parse, the model has a native recovery path.

  - "native": single channel. Same wire schemas, but the compact prompt
    tells the model to issue ALL actions as native tool calls and avoid
    text action blocks.

Notes:
  - The core never probes server capability (transport-agnostic by design).
    Set backend.supports_tools yourself if your host detects it; the merge
    catch for unsolicited native calls stays on in EVERY mode regardless.
  - Schemas come from the tools' flat `parameters` declarations
    (tools_schemas()). Tools without `parameters` stay prose-only and are
    skipped from the wire — still reachable via **TOOL:** blocks.
  - Requires a function-calling-capable model (e.g. qwen2.5/x/llama-class
    or any OpenAI-tools-compatible endpoint). Non-tool models just never
    emit tool_calls; the session degrades gracefully.
"""

from agentic_core import AgenticSession, AgenticInterrupted
from agentic_core.backends import OpenAIBackend
from agentic_core.tools import tools_schemas, tools_native_prompt, register
from agentic_core.skills import Skill, register as register_skill


class GreetingSkill(Skill):
    name = "greeting"
    description = "Generate a greeting."
    system_prompt = "When invoked, produce a friendly greeting for the user."
    parameters = {"name": {"type": "string", "required": False,
                           "description": "Person to greet"}}

    def execute(self, params, workdir=None, session=None):
        return f"[Skill greeting invoked] Hello, {params.get('name', 'friend')}!"


def main():
    register_skill(GreetingSkill())

    # Tools without flat `parameters` (none among the 16 built-ins today)
    # would be skipped from schemas but stay reachable via TOOL: blocks.
    schemas = tools_schemas()
    print(f"[native] {len(schemas)} tool schemas on the wire\n")

    backend = OpenAIBackend(
        base_url="http://localhost:11434",   # Ollama /v1, vLLM, LM Studio, ...
        api_key="ollama",
        model="glm-5.3-flash:cloud",         # your function-calling model
        ctx_size=32768,
        tools_schemas=schemas,               # ← solicitation: tools param
    )
    # Optional host-side capability hint (the core never probes the server):
    # backend.supports_tools = True

    session = AgenticSession(
        backend=backend,
        workdir=".",
        tools=True,
        skills=True,
        name="myapp",
        native_tools=True,     # True = solicitation on (schemas + native prompt)
                               # None/False = passive capture only (text mode)
        tools_mode="hybrid",   # "hybrid" = schemas + text blocks preferred,
                               #   file tools as native backup (recommended)
                               # "native" = schemas, single channel
                               # "text" = no schemas (default, passive capture stays)
        auto_approve_safe=False,
        max_feedback_rounds=3,
    )
    # What hybrid changed, for the curious:
    #   - prompt: per-tool prose listing replaced by tools_native_prompt(
    #     "hybrid") — the one-channel rule + text-block preference
    #   - wire: {"tools": [...]} attached in backend.call()
    #   - history: native calls round-trip as assistant(tool_calls) +
    #     tool-role results (synthesized ids at the wire boundary only)
    #   - channel mixing (a text block poured into a tool call) is detected
    #     (name-slot marker guard) and steered — 3rd event suggests going
    #     native-only for the session

    print("Agentic session ready (native hybrid mode). Type 'quit' to exit.\n")
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
            continue
        print()


if __name__ == "__main__":
    main()
