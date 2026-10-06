"""MCP (Model Context Protocol) client — discovers and invokes tools
from MCP servers.

Host-agnostic by design (MCP-ARCH.md upstream extraction):
- config: the host passes a mcpServers dict (load it from wherever)
- confinement: optional path_checker(path, workdir) -> error|None hook
  for output-file saves (hosts without confinement pass None)
- lifecycle: the host decides WHEN to connect (boot vs lazy-on-demand);
  this module only provides connect()/sync()/close_all()

Three transports:
- SSE: HTTP Server-Sent Events (url ending with /sse)
- Streamable HTTP: direct JSON-RPC POST (url not ending with /sse)
- stdio: subprocess with newline-delimited JSON-RPC

httpx is optional (like the http_request tool): stdio works without it;
HTTP transports raise a clear install hint.
"""

from .manager import MCPManager, MCPTool, MCPUsageTool, _prefixed_name

__all__ = ["MCPManager", "MCPTool", "MCPUsageTool", "_prefixed_name"]
