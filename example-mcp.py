"""Example: MCP (Model Context Protocol) integration with agentic-core.

Run from the agentic-core repo root:

    python example-mcp.py

Demonstrates the full host wiring:
  1. MCPManager with a config source (the mcpServers dict — in a real
     app, load it from your config file, fresh on every call)
  2. Lazy activation: the mcp_usage meta-tool + a stub line in the
     system prompt (MCP tools cost zero prompt tokens until used)
  3. Lifecycle: connect on demand, sync, close at shutdown

The config below mirrors a typical .mercury.json mcpServers section:
an HTTP server (huggingface), a stdio server (mcp-mermaid), and a
disabled one (playwright) that is skipped.
"""

import json

from agentic_core import AgenticSession, build_system_prompt
from agentic_core.mcp import MCPManager, MCPUsageTool
from agentic_core.tools import register

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

    # 2. Lazy activation: register the meta-tool; MCP tool docs are
    #    served on demand (mcp_usage connects + returns the docs).
    register(MCPUsageTool(mcp))

    stub = "MCP TOOLS AVAILABLE (not shown here): huggingface, mcp-mermaid.\n" \
           "Invoke TOOL:mcp_usage to connect and see their full " \
           "documentation when the user wants to use them."

    # 3. The session — the stub line rides in extra_sections
    from ollama import Client
    session = AgenticSession(
        client=Client(host="http://localhost:11434"),
        model="glm-5.3-flash:cloud",
        workdir=".",
        system_prompt=build_system_prompt(
            name="mcp-demo", tools=True, skills=False,
            extra_sections=[stub],
        ),
        auto_approve_safe=True,   # demo: no interactive confirmations
    )

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
