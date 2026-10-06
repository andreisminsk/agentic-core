"""MCP client tests (MCP-ARCH.md §5): config parsing, prefixing,
registration, enablement gating, output confinement, timeout,
reconnect. Mock transports only — no real servers, no network."""

import json
import threading

import pytest

import agentic_core.tools as tools_mod
from agentic_core.mcp import (
    MCPManager,
    MCPTool,
    MCPUsageTool,
    _prefixed_name,
)


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(tools_mod._registry)
    tools_mod._registry.clear()
    yield
    tools_mod._registry.clear()
    tools_mod._registry.update(saved)


class FakeTransport:
    """Canned transport: returns fixed tools/list and tools/call."""

    def __init__(self, tools=None, call_result=None, fail_call=False,
                 name="fake"):
        self.tools = tools or []
        self.call_result = call_result or {"content": [{"type": "text",
                                                        "text": "ok"}]}
        self.fail_call = fail_call
        self.name = name
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True
        return {}

    def list_tools(self):
        return self.tools

    def call_tool(self, tool_name, arguments):
        if self.fail_call:
            raise RuntimeError("server exploded")
        return self.call_result

    def close(self):
        self.closed = True


def _tool_def(name="search", desc="Search things", props=None, required=None):
    return {
        "name": name,
        "description": desc,
        "inputSchema": {
            "type": "object",
            "properties": props or {"query": {"type": "string",
                                              "description": "the query"}},
            "required": required or ["query"],
        },
    }


# ── Naming ──────────────────────────────────────────────────────────────

class TestPrefixing:
    def test_basic(self):
        assert _prefixed_name("context7", "resolve") == "mcp_context7_resolve"

    def test_strips_mcp_prefix(self):
        assert _prefixed_name("mcp-weather", "get") == "mcp_weather_get"

    def test_hyphens_to_underscores(self):
        assert _prefixed_name("my-server", "do-thing") == "mcp_my_server_do_thing"


# ── MCPTool wrapper ─────────────────────────────────────────────────────

class TestMCPTool:
    def test_prompt_from_schema(self):
        t = MCPTool(name="mcp_s_search", description="Search things",
                    input_schema=_tool_def()["inputSchema"],
                    server_name="s", transport=FakeTransport())
        assert "## mcp_s_search" in t.system_prompt
        assert "query (string, required)" in t.system_prompt

    def test_call_returns_text(self, tmp_path):
        t = MCPTool(name="mcp_s_search", description="d",
                    input_schema={}, server_name="s",
                    transport=FakeTransport(call_result={
                        "content": [{"type": "text", "text": "hello"}]}))
        r = t.execute({"query": "x"}, str(tmp_path))
        assert r == "hello"

    def test_call_error_observation(self, tmp_path):
        t = MCPTool(name="mcp_s_search", description="d",
                    input_schema={}, server_name="s",
                    transport=FakeTransport(fail_call=True))
        r = t.execute({}, str(tmp_path))
        assert "[MCP tool error:" in r

    def test_large_output_saved_confined(self, tmp_path):
        """D7: >6000 chars → file in the workspace via check_path."""
        big = "x" * 7000
        t = MCPTool(name="mcp_s_search", description="d",
                    input_schema={}, server_name="s",
                    transport=FakeTransport(call_result={
                        "content": [{"type": "text", "text": big}]}))
        r = t.execute({}, str(tmp_path))
        assert "Output saved to:" in r
        saved = tmp_path / "mcp_s_search_output.txt"
        assert saved.is_file()
        assert saved.read_text(encoding="utf-8") == big

    def test_binary_saved(self, tmp_path):
        import base64
        payload = base64.b64encode(b"pngdata").decode()
        t = MCPTool(name="mcp_s_search", description="d",
                    input_schema={}, server_name="s",
                    transport=FakeTransport(call_result={
                        "content": [{"type": "image", "mimeType": "image/png",
                                     "data": payload}]}))
        r = t.execute({}, str(tmp_path))
        assert "Image saved to:" in r
        assert (tmp_path / "mcp_s_search_0.png").read_bytes() == b"pngdata"

    def test_auto_approve_default_false(self):
        t = MCPTool(name="mcp_s_x", description="d", input_schema={},
                    server_name="s", transport=FakeTransport())
        assert t.auto_approve is False


# ── Manager ─────────────────────────────────────────────────────────────

class TestManager:
    def test_register_tools_into_registry(self):
        m = MCPManager(get_config=lambda: {})
        m.tools = [MCPTool(name="mcp_s_search", description="d",
                           input_schema={}, server_name="s",
                           transport=FakeTransport())]
        count = m.register_tools()
        assert count == 1
        from agentic_core.tools import get
        assert get("mcp_s_search") is not None

    def test_register_idempotent(self):
        m = MCPManager(get_config=lambda: {})
        m.tools = [MCPTool(name="mcp_s_search", description="d",
                           input_schema={}, server_name="s",
                           transport=FakeTransport())]
        assert m.register_tools() == 1
        assert m.register_tools() == 0  # already registered

    def test_refresh_tools_cap(self):
        """D6: >20 tools from one server → only 20 registered."""
        m = MCPManager(get_config=lambda: {"big": {}})
        fake = FakeTransport(tools=[_tool_def(name=f"t{i}") for i in range(30)])
        m.transports["big"] = fake
        m._refresh_tools()
        assert len(m.tools) == 20

    def test_sync_drops_disabled(self):
        m = MCPManager(get_config=lambda: {"srv": {"enabled": False}})
        fake = FakeTransport()
        m.transports["srv"] = fake
        m.sync()
        assert "srv" not in m.transports
        assert fake.closed

    def test_sync_reconnects_dead(self, monkeypatch):
        """A dead transport is replaced by a fresh connection."""
        m = MCPManager(get_config=lambda: {"srv": {"url": "http://x"}})
        dead = FakeTransport()
        dead._alive = lambda: False   # simulate a dead subprocess
        m.transports["srv"] = dead
        created = FakeTransport(tools=[_tool_def()])
        monkeypatch.setattr(m, "_create_transport", lambda name, cfg: created)
        m.sync()
        assert m.transports["srv"] is created
        assert created.connected

    def test_close_all(self):
        m = MCPManager(get_config=lambda: {})
        fake = FakeTransport()
        m.transports["srv"] = fake
        m.close_all()
        assert fake.closed
        assert m.transports == {}
        assert m.tools == []


# ── Config resolution ───────────────────────────────────────────────────

class TestConfig:
    def test_auth_token_env_escape(self, monkeypatch):
        """D2: auth_token_env reads the token from the environment."""
        monkeypatch.setenv("MY_MCP_TOKEN", "secret123")
        m = MCPManager(get_config=lambda: {})
        cfg = {"url": "http://x/sse", "auth_token_env": "MY_MCP_TOKEN"}
        t = m._create_transport("srv", cfg)
        assert t.auth_token == "secret123"

    def test_inline_auth_token(self):
        m = MCPManager(get_config=lambda: {})
        t = m._create_transport("srv", {"url": "http://x/sse",
                                        "auth_token": "inline"})
        assert t.auth_token == "inline"

    def test_transport_selection_sse(self):
        from agentic_core.mcp.transports import SSETransport
        m = MCPManager(get_config=lambda: {})
        t = m._create_transport("srv", {"url": "http://x/sse"})
        assert isinstance(t, SSETransport)

    def test_transport_selection_http(self):
        from agentic_core.mcp.transports import StreamableHTTPTransport
        m = MCPManager(get_config=lambda: {})
        t = m._create_transport("srv", {"url": "http://x/mcp"})
        assert isinstance(t, StreamableHTTPTransport)

    def test_transport_selection_stdio(self):
        from agentic_core.mcp.transports import StdioTransport
        m = MCPManager(get_config=lambda: {})
        t = m._create_transport("srv", {"command": "npx", "args": ["-y", "x"]})
        assert isinstance(t, StdioTransport)

    def test_no_url_or_command(self):
        m = MCPManager(get_config=lambda: {})
        assert m._create_transport("srv", {}) is None

    def test_auth_token_env_file_fallback(self, tmp_path, monkeypatch):
        """auth_token_env falls back to the host-named env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_MCP_TOKEN=filetoken\n", encoding="utf-8")
        monkeypatch.delenv("MY_MCP_TOKEN", raising=False)
        m = MCPManager(get_config=lambda: {}, env_file=str(env_file))
        cfg = {"url": "http://x/sse", "auth_token_env": "MY_MCP_TOKEN"}
        t = m._create_transport("srv", cfg)
        assert t.auth_token == "filetoken"


# ── Lazy activation (MCP-ARCH.md L1–L4) ─────────────────────────────────

class TestLazyActivation:
    def _manager_with_tools(self):
        m = MCPManager(get_config=lambda: {"ctx": {}})
        # Simulate the connected state: transports present so the lazy
        # connect() inside mcp_usage.execute is a no-op, tools populated.
        m.transports["ctx"] = FakeTransport(
            tools=[_tool_def(),
                   _tool_def(name="get", desc="Get ctx")])
        m._refresh_tools()
        return m

    def test_mcp_tools_excluded_from_prompt(self):
        """L2: registered mcp_* tools never appear in tools_system_prompt."""
        from agentic_core.tools import register, tools_system_prompt
        m = self._manager_with_tools()
        for t in m.tools:
            register(t)
        register(MCPUsageTool(m))
        prompt = tools_system_prompt()
        assert "mcp_ctx_search" not in prompt      # excluded
        assert "mcp_ctx_get" not in prompt
        assert "## mcp_usage" in prompt              # meta-tool IS shown

    def test_non_mcp_tools_still_prompted(self):
        """The filter must not touch regular tools."""
        from agentic_core.tools import register, tools_system_prompt
        m = self._manager_with_tools()
        register(MCPUsageTool(m))
        prompt = tools_system_prompt()
        assert "## mcp_usage" in prompt

    def test_usage_serves_full_docs(self):
        """L1: mcp_usage returns the generated docs as the observation."""
        m = self._manager_with_tools()
        usage = MCPUsageTool(m)
        r = usage.execute({"server": "ctx"})
        assert "MCP server 'ctx' — 2 tool(s)" in r
        assert "## mcp_ctx_search" in r
        assert "query (string, required)" in r

    def test_usage_all_servers(self):
        m = self._manager_with_tools()
        r = MCPUsageTool(m).execute({})
        assert "MCP server 'ctx'" in r

    def test_usage_list_only(self):
        m = self._manager_with_tools()
        r = MCPUsageTool(m).execute({"list_only": True})
        assert "- mcp_ctx_search: Search things" in r
        assert "- mcp_ctx_get: Get ctx" in r
        assert "(string, required)" not in r   # no full docs

    def test_usage_unknown_server(self):
        m = self._manager_with_tools()
        r = MCPUsageTool(m).execute({"server": "ghost"})
        assert "No MCP server named" in r
        assert "ctx" in r                        # suggests available

    def test_usage_no_tools(self):
        m = MCPManager(get_config=lambda: {})
        r = MCPUsageTool(m).execute({})
        assert "No MCP tools available" in r

    def test_usage_auto_approve(self):
        """L4: read-only meta-tool — no confirmation."""
        m = self._manager_with_tools()
        assert MCPUsageTool(m).auto_approve is True

    def test_connect_registers_tools(self, monkeypatch):
        """Regression: the lazy flow registers tools at connect time —
        mcp_usage connects mid-session; without registration there,
        every subsequent TOOL block fails with 'unknown tool'."""
        from agentic_core.tools import get as get_tool
        m = MCPManager(get_config=lambda: {"ctx": {"url": "http://x"}})
        fake = FakeTransport(tools=[_tool_def(name="search")])
        monkeypatch.setattr(m, "_create_transport", lambda name, cfg: fake)
        m.connect("ctx")
        assert get_tool("mcp_ctx_search") is not None   # in the registry

    def test_register_respects_enablement(self):
        """Late registration applies the host's enablement gate —
        disabled mcp_* names are not registered."""
        from agentic_core.tools import get as get_tool
        m = self._manager_with_tools()
        m.register_tools(
            enabled_check=lambda name: name != "mcp_ctx_get")
        assert get_tool("mcp_ctx_search") is not None
        assert get_tool("mcp_ctx_get") is None      # gated
