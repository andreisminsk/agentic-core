"""MCP manager, tool wrapper, and meta-tool — host-agnostic core."""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import threading
import uuid

logger = logging.getLogger(__name__)

MAX_TOOLS_PER_SERVER = 20
LARGE_OUTPUT_THRESHOLD = 6000

_MIME_EXT = {
    "image/svg+xml": ".svg", "text/html": ".html", "text/xml": ".xml",
    "application/json": ".json", "text/csv": ".csv",
    "text/markdown": ".md", "text/plain": ".txt",
}


def _prefixed_name(server_name, tool_name):
    clean = server_name.removeprefix("mcp-").removeprefix("mcp_")
    return f"mcp_{clean.replace('-', '_')}_{tool_name.replace('-', '_')}"


class MCPTool:
    """A tool backed by an MCP server tool.

    path_checker: optional (path, workdir) -> error-string-or-None;
    when set, output-file saves are refused outside the allowed roots.
    """

    def __init__(self, name, description, input_schema, server_name,
                 transport, auto_approve=False, original_name=None,
                 path_checker=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}
        self.system_prompt = self._build_system_prompt()
        self.parameters = self._build_parameters()
        self.server_name = server_name
        self.transport = transport
        self.auto_approve = auto_approve
        self.original_name = original_name or name
        self.path_checker = path_checker

    def _build_system_prompt(self):
        props = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        lines = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            req = "required" if pname in required else "optional"
            pdesc = pdef.get("description", "")
            lines.append(f"- {pname} ({ptype}, {req}): {pdesc}")
        parts = [f"## {self.name}", self.description]
        if lines:
            parts += ["", "Parameters (JSON object):"] + lines
        return "\n".join(parts)

    def _build_parameters(self):
        params = {}
        props = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        for pname, pdef in props.items():
            params[pname] = {
                "type": pdef.get("type", "string"),
                "required": pname in required,
                "description": pdef.get("description", ""),
            }
        return params

    @staticmethod
    def _detect_extension(text, mime_type=""):
        if mime_type:
            ext = _MIME_EXT.get(mime_type)
            if ext:
                return ext
        stripped = text.lstrip()
        if stripped.startswith("<svg"):
            return ".svg"
        if stripped.startswith(("<!DOCTYPE html", "<html")):
            return ".html"
        if stripped.startswith(("{", "[")):
            return ".json"
        return ".txt"

    def _confined_save(self, workdir, filename, data, binary=False):
        if self.path_checker is not None:
            err = self.path_checker(filename, workdir)
            if err:
                return None, err
        path = os.path.join(workdir, filename)
        try:
            if binary:
                with open(path, "wb") as f:
                    f.write(data)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(data)
            return path, None
        except OSError as e:
            return None, str(e)

    def execute(self, params, workdir=None, session=None):
        try:
            result = self.transport.call_tool(
                tool_name=self.original_name, arguments=params or {})
            return self._format_result(result, workdir)
        except Exception as e:
            return f"[MCP tool error: {e}]"

    def _format_result(self, result, workdir):
        if not isinstance(result, dict):
            return str(result)
        content = result.get("content", [])
        if not content:
            return json.dumps(result, indent=2)
        parts = []
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                if isinstance(item, str):
                    parts.append(self._text_part(item, workdir))
                continue
            mime = item.get("mimeType", "")
            data = item.get("data", "")
            if mime.startswith("image/") or mime == "application/octet-stream":
                ext = mime.split("/")[-1].replace("jpeg", "jpg")
                if ext == "plain":
                    ext = "txt"
                filename = f"{self.name}_{i}.{ext}"
                try:
                    binary = base64.b64decode(data)
                    path, err = self._confined_save(workdir or ".", filename,
                                                    binary, binary=True)
                    if err:
                        parts.append(f"[Binary {mime}: save failed — {err}]")
                    else:
                        parts.append(f"Image saved to: {path} ({len(binary)} bytes)")
                except Exception as e:
                    parts.append(f"[Binary {mime}: decode failed — {e}]")
            else:
                text = item.get("text", "")
                if text:
                    parts.append(self._text_part(text, workdir, mime))
        return "\n".join(parts) if parts else json.dumps(result, indent=2)

    def _text_part(self, text, workdir, mime=""):
        if len(text) <= LARGE_OUTPUT_THRESHOLD or not workdir:
            return text
        ext = self._detect_extension(text, mime)
        filename = f"{self.name}_output{ext}"
        path, err = self._confined_save(workdir, filename, text)
        if err:
            return text
        return (f"Output saved to: {path} ({len(text)} chars). "
                f"Read it with read_file.")


class MCPUsageTool:
    """Lazy-activation meta-tool: serves MCP tool docs on demand."""

    name = "mcp_usage"
    description = ("List MCP servers and show the full documentation of "
                    "their tools — invoke when the user wants to use MCP "
                    "tools.")
    auto_approve = True

    system_prompt = (
        "## mcp_usage\n"
        "MCP tools are available but not shown here — their full docs "
        "are served on demand.\n"
        "Parameters (JSON object):\n"
        "- server (string, optional): one server's name — omit to see ALL\n"
        "- list_only (boolean, optional, default false): names and "
        "descriptions only\n"
        "Use when the user asks to use MCP tools or asks what MCP tools "
        "exist. The observation contains the tools' full documentation — "
        "then invoke them like any other tool."
    )
    parameters = {
        "server": {"type": "string", "required": False,
                   "description": "One server's name; omit for all"},
        "list_only": {"type": "boolean", "required": False,
                      "description": "Names/descriptions only (default false)"},
    }

    def __init__(self, manager):
        self.manager = manager

    def execute(self, params, workdir=None, session=None):
        server = str(params.get("server") or "").strip()
        list_only = bool(params.get("list_only", False))
        self.manager.connect(server if server else None)
        tools = self.manager.tools
        if not tools:
            return ("[No MCP tools available — no servers connected "
                    "successfully. Check the MCP server configuration.]")
        by_server = {}
        for t in tools:
            by_server.setdefault(t.server_name, []).append(t)
        if server:
            group = by_server.get(server)
            if group is None:
                avail = ", ".join(sorted(by_server))
                return (f"[No MCP server named {server!r} — available: {avail}]")
            return self._docs(server, group, list_only)
        return "\n\n".join(self._docs(n, by_server[n], list_only)
                           for n in sorted(by_server))

    @staticmethod
    def _docs(server, group, list_only):
        header = f"MCP server '{server}' — {len(group)} tool(s):"
        if list_only:
            return "\n".join([header] +
                             [f"- {t.name}: {t.description}" for t in group])
        blocks = [header, ""]
        for t in group:
            blocks.append(t.system_prompt)
            blocks.append("")
        return "\n".join(blocks)


class MCPManager:
    """Connects MCP servers, discovers and registers their tools.

    Host injections:
    - get_config() -> the mcpServers dict (host loads it from its own
      config source; called fresh on every connect/sync)
    - path_checker: optional (path, workdir) -> error|None for output
      saves
    - resolve_auth_token(cfg) -> token|None: default reads auth_token /
      auth_token_env (env, then a .env-style file the host names)
    """

    def __init__(self, get_config, path_checker=None,
                 resolve_auth_token=None, env_file=None):
        self.get_config = get_config
        self.path_checker = path_checker
        self.transports = {}
        self.tools = []
        self._lock = threading.Lock()
        if resolve_auth_token is not None:
            self._resolve_auth_token = resolve_auth_token
        self._env_file = env_file

    def _resolve_auth_token(self, cfg):
        token = cfg.get("auth_token")
        if token:
            return token
        env_name = cfg.get("auth_token_env")
        if not env_name:
            return None
        token = os.environ.get(env_name)
        if token:
            return token
        if self._env_file:
            try:
                with open(self._env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(env_name + "="):
                            return line.split("=", 1)[1].strip() or None
            except OSError:
                pass
        return None

    def _create_transport(self, name, cfg):
        from .transports import SSETransport, StreamableHTTPTransport, StdioTransport
        timeout = int(cfg.get("timeout", 120) or 120)
        if "url" in cfg:
            if cfg["url"].rstrip("/").endswith("/sse"):
                return SSETransport(
                    url=cfg["url"], name=name, timeout=timeout,
                    auth_token=self._resolve_auth_token(cfg),
                    headers=cfg.get("headers"))
            return StreamableHTTPTransport(
                url=cfg["url"], name=name, timeout=timeout,
                auth_token=self._resolve_auth_token(cfg),
                headers=cfg.get("headers"))
        if "command" in cfg:
            return StdioTransport(
                command=cfg["command"], args=cfg.get("args", []),
                env=cfg.get("env", {}), name=name, timeout=timeout)
        return None

    def connect(self, server_name=None):
        """Connect servers on demand. server_name=None → all enabled,
        in parallel. Failures are logged, never raised."""
        configs = self.get_config() or {}
        if server_name is not None:
            configs = {server_name: configs[server_name]} \
                if server_name in configs else {}
        errors = {}
        threads = []

        def _connect(name, cfg):
            try:
                transport = self._create_transport(name, cfg)
                if transport is None:
                    errors[name] = "no 'url' or 'command' field"
                    return
                transport.connect()
                with self._lock:
                    self.transports[name] = transport
            except Exception as e:
                errors[name] = str(e)

        for name, cfg in configs.items():
            if cfg.get("enabled", True) is False:
                logger.info("[MCP] Skipping %s (disabled)", name)
                continue
            if name in self.transports:
                continue
            t = threading.Thread(target=_connect, args=(name, cfg),
                                 daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=60)
        for name, error in errors.items():
            logger.warning("[MCP] Failed to connect to %s: %s", name, error)
        self._refresh_tools()
        self.register_tools()
        return list(self.transports.keys())

    def sync(self):
        """Drop removed/disabled servers; reconnect dead-but-connected.
        Never first-connects (lazy hosts call connect() on demand)."""
        configs = self.get_config() or {}
        for name in list(self.transports.keys()):
            cfg = configs.get(name)
            if cfg is None or cfg.get("enabled", True) is False:
                try:
                    self.transports.pop(name).close()
                except Exception:
                    pass
        for name in list(self.transports.keys()):
            existing = self.transports.get(name)
            alive = getattr(existing, "_alive", lambda: True)()
            if not alive:
                cfg = configs.get(name, {})
                try:
                    self.transports.pop(name).close()
                    transport = self._create_transport(name, cfg)
                    if transport is not None:
                        transport.connect()
                        with self._lock:
                            self.transports[name] = transport
                except Exception as e:
                    logger.warning("[MCP] Reconnect failed for %s: %s",
                                   name, e)
        self._refresh_tools()
        self.register_tools()

    def _refresh_tools(self):
        self.tools = []
        configs = self.get_config() or {}
        for server_name, transport in self.transports.items():
            cfg = configs.get(server_name, {})
            try:
                tool_defs = transport.list_tools()
            except Exception as e:
                logger.warning("[MCP] tools/list failed for %s: %s",
                               server_name, e)
                continue
            if len(tool_defs) > MAX_TOOLS_PER_SERVER:
                logger.warning("[MCP] %s exposes %d tools — registering "
                               "only the first %d (cap)",
                               server_name, len(tool_defs),
                               MAX_TOOLS_PER_SERVER)
                tool_defs = tool_defs[:MAX_TOOLS_PER_SERVER]
            for td in tool_defs:
                original = td.get("name", "")
                if not original:
                    continue
                self.tools.append(MCPTool(
                    name=_prefixed_name(server_name, original),
                    description=td.get("description", ""),
                    input_schema=td.get("inputSchema", {}),
                    server_name=server_name,
                    transport=transport,
                    auto_approve=bool(cfg.get("auto_approve", False)),
                    original_name=original,
                    path_checker=self.path_checker,
                ))

    def register_tools(self, enabled_check=None):
        """Register tools into the agentic-core registry.

        enabled_check(name) -> bool: the host's enablement gate
        (e.g. .mercury.json tools section). None = all enabled."""
        from ..tools import _registry, register
        count = 0
        for tool in self.tools:
            if tool.name in _registry:
                continue
            if enabled_check is not None and not enabled_check(tool.name):
                continue
            register(tool)
            count += 1
        return count

    def close_all(self):
        for transport in self.transports.values():
            try:
                transport.close()
            except Exception:
                pass
        self.transports.clear()
        self.tools.clear()
