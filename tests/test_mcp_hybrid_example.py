"""Verify example-mcp.py's hybrid wiring against core invariants.

All checks run inside test functions — module-level side effects (the
first version executed checks at import, and sys.exit during collection
killed the whole pytest run).
"""
import ast
import sys

SRC_PATH = "example-mcp.py"


def _source():
    with open(SRC_PATH) as f:
        return f.read()


def _ordered_call_names(tree):
    """(name, lineno) for every call, in source order."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", None) or getattr(n.func, "attr", "")
            out.append((name, n.lineno))
    out.sort(key=lambda t: t[1])
    return out


def test_example_declares_hybrid_wiring():
    src = _source()
    assert 'tools_mode="hybrid"' in src, "session must run tools_mode='hybrid'"
    assert "native_tools=True" in src, "session must set native_tools=True"
    assert "tools_schemas()" in src, "schemas must be attached (tools array)"
    assert "from agentic_core.backends import OpenAIBackend" in src, \
        "hybrid requires an OpenAI-compatible backend"


def test_no_stale_text_mode_imports():
    src = _source()
    assert "build_system_prompt(" not in src, \
        "session should build its own native prompt, not a custom stub"
    assert "from ollama import Client" not in src, \
        "native solicitation goes through OpenAIBackend, not raw ollama Client"


def test_lifecycle_ordering_register_connect_then_schemas():
    src = _source()
    tree = ast.parse(src)
    calls = _ordered_call_names(tree)

    def first_idx(name):
        for i, (n, _) in enumerate(calls):
            if n == name:
                return i
        return -1

    reg = first_idx("register")
    conn = first_idx("connect")
    schemas = first_idx("tools_schemas")
    assert schemas > -1, "tools_schemas must be called"
    assert -1 < reg < schemas, "mcp_usage registered before schemas attach"
    assert -1 < conn < schemas, "mcp.connect() before schemas attach"


def test_core_hybrid_prompt_honors_contract():
    from agentic_core.tools import tools_native_prompt
    p = tools_native_prompt("hybrid")
    assert "PREFER text action blocks" in p, \
        "core hybrid prompt must prefer text blocks for file content"
    assert "read_file, write_file, replace_in_file" in p, \
        "core hybrid prompt must name the native backup file tools"
    assert "native tool calls" in p


def test_mcp_tool_qualifies_for_wire():
    from agentic_core.mcp.manager import MCPTool
    from agentic_core.tools import _flat_params_to_schema, tools_schemas
    t = MCPTool(name="mcp_huggingface_search", description="d",
                input_schema={"properties": {"q": {"type": "string"}},
                              "required": ["q"]},
                server_name="huggingface", transport=None)
    flat = _flat_params_to_schema(t.name, t.description, t.parameters)
    assert flat["function"]["parameters"]["required"] == ["q"]

    # Built-in registry (fresh process): every schema is well-formed wire.
    schemas = tools_schemas()
    assert schemas, "built-in tools must produce schemas"
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["name"]
    names = [s["function"]["name"] for s in schemas]
    # mcp_usage is host-registered (the example does it), not a built-in —
    # here we just report; the ordering test proves the example registers
    # it before schemas are built.
    print("wire names:", names)
    assert "read_file" in names or "write_file" in names, \
        "file tools stay on the wire as native backup in hybrid mode"
