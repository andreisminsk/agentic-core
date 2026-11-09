"""Example: MCP (Model Context Protocol) integration with agentic-core,
running in NATIVE HYBRID mode.

Run from the agentic-core repo root:

    python example-mcp.py

Demonstrates the full host wiring:
  1. MCPManager with a config source (the mcpServers dict — in a real
     app, load it from your config file, fresh on every call)
  2. Native hybrid mode: tool schemas (BUILT-IN + MCP) are attached to
     the LLM request, so function-calling models emit structured
     tool_calls; the prompt still PREFERs text action blocks
     (FILE:/WRITE:/EDIT:) for file content, with read_file/write_file/
     replace_in_file kept on the wire as BACKUP.
  3. Lifecycle: connect at startup (so MCP schemas travel on the wire),
     mcp_usage stays registered as the lazy re-activation path, close
     at shutdown.

The config below mirrors a typical .mercury.json mcpServers section:
an HTTP server (huggingface), a stdio server (mcp-mermaid), and a
disabled one (playwright) that is skipped.
"""

from agentic_core import AgenticSession
from agentic_core.backends import OpenAIBackend
from agentic_core.mcp import MCPManager, MCPUsageTool
from agentic_core.tools import tools_schemas, register

MCP_SERVERS = {
    "huggingface": {
        "enabled": True,
        "url": "https://huggingface.co/mcp",
        "timeout": 120,
        "auto_approve": True,
    },
    "mcp-mermaid": {
        "enabled": True,
        "command": "mcp-mermaid",
        "timeout": 120,
    },
    "playwright": {
        "enabled": True,          # skipped by connect()
        "command": "npx.cmd",
        "args": ["-y", "@playwright/mcp@latest"],
        "timeout": 180,
        "auto_approve": False,
    },
}


def main():
    # 1. The manager — host-injected config source
    mcp = MCPManager(get_config=lambda: MCP_SERVERS)

    # 2. Register the meta-tool FIRST: it stays on the wire in hybrid
    #    mode (it declares `parameters`) as the re-activation path for
    #    servers not connected yet / tools registered later.
    register(MCPUsageTool(mcp))

    # 3. Eager connect BEFORE building schemas — in native hybrid mode
    #    registered mcp_* tools travel AS SCHEMAS on the wire (the
    #    prompt skip in tools_system_prompt serves docs on demand, but
    #    native-calling models never read prompt docs — their channel
    #    is the tools array). MCPTool qualifies: its `parameters` come
    #    from the server-declared input_schema.
    mcp.connect()

    # 4. The session — solicited native tools, hybrid mode.
    #    No custom system_prompt: the session builds it and, because
    #    native_tools + hybrid are set, REPLACES the per-tool prose
    #    listing with the compact channel policy
    #    (tools_native_prompt("hybrid")): native calls preferred,
    #    text action blocks for file content, native file tools as
    #    backup. Schemas are attached to the backend automatically.
    session = AgenticSession(
        backend=OpenAIBackend(
            base_url="http://localhost:11434",   # Ollama /v1, vLLM, LM Studio, ...
            api_key="ollama",
            model="glm-5.3-flash:cloud",         # your function-calling model
            ctx_size=32768,
            tools_schemas=tools_schemas(),       # built-ins + mcp_* tools
        ),
        workdir=".",
        name="mcp-demo",
        tools=True,
        skills=False,
        native_tools=True,     # solicitation on (schemas + native prompt)
        tools_mode="hybrid",   # schemas + text blocks preferred for files,
                               #   read_file/write_file/replace_in_file as
                               #   native backup; "native" = single channel
        auto_approve_safe=True,   # demo: no interactive confirmations
    )
    # LAZY alternative: skip mcp.connect(), then after mcp_usage connects
    # the host must reattach the refreshed schemas —
    #   session.backend.tools_schemas = tools_schemas()
    # (mcp_usage still returns full docs; newly registered mcp_* tools
    # become callable natively on the next model call.)

    print("Agentic session ready (native HYBRID mode: native tool_calls, "
          "text blocks for file content).\n")
    print("Ask things like: 'search huggingface for text-to-speech models'")
    print("or: 'what MCP tools do we have?'  (Ctrl+C to quit)\n")
    try:
        while True:
            user = input("you: ").strip()
            if not user:
                continue
            if user in ("quit", "exit"):
                break
            response = session.run(user)
            print(f"agent: {response}\n")
    except KeyboardInterrupt:
        print("\n[bye]")
    finally:
        mcp.close_all()


if __name__ == "__main__":
    main()
