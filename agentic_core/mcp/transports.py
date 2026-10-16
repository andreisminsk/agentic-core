"""MCP transports: SSE, Streamable HTTP, stdio.

httpx is optional (the http_request-tool precedent): stdio works
without it; the HTTP transports raise a clear install hint.

Error taxonomy (ported from uhu 1.18.1): MCPConnectionError marks a
transport failure a reconnect may fix; MCPError marks a JSON-RPC error
returned by the server — the transport is alive, retrying won't help.
MCPTool._invoke_with_reconnect (manager.py) reconnects once on the
former and never on the latter. Server-initiated messages (pings,
notifications, requests) are answered or ignored inline — the client
implements no server→client features, and such messages must never be
mistaken for pending responses.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import uuid

logger = logging.getLogger(__name__)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def _require_httpx():
    if not HAS_HTTPX:
        raise ImportError("httpx is required for HTTP MCP transports. "
                          "Install it with: pip install httpx")


class MCPError(RuntimeError):
    """JSON-RPC error returned by an MCP server — reconnecting won't help."""


class MCPConnectionError(RuntimeError):
    """Transport/connection failure — a reconnect may restore the session."""


class SSETransport:
    """MCP over HTTP SSE (url ending with /sse)."""

    def __init__(self, url, name, timeout=120, auth_token=None, headers=None):
        _require_httpx()
        self.url = url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.auth_token = auth_token
        self.extra_headers = headers or {}
        self.message_url = None
        self._client = None
        self._response_queue = None
        self._endpoint_event = None
        self._reader = None

    def connect(self):
        import queue
        _require_httpx()
        self._response_queue = queue.Queue()
        self._endpoint_event = threading.Event()
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        h.update(self.extra_headers)
        self._client = httpx.Client(headers=h, timeout=self.timeout)

        sse_url = self.url if self.url.endswith("/sse") else self.url + "/sse"
        try:
            resp = self._client.get(sse_url,
                                    headers={"Accept": "text/event-stream"})
            resp.raise_for_status()
        except Exception as e:
            raise MCPConnectionError(
                f"MCP SSE: failed to connect to {sse_url}: {e}")
        self._sse_resp = resp

        self._reader = threading.Thread(target=self._sse_reader, daemon=True,
                                        name=f"mcp-sse-{self.name}")
        self._reader.start()

        if not self._endpoint_event.wait(timeout=30):
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.url)
            base = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
            self.message_url = f"{base}/messages/"
            logger.warning("[MCP] endpoint discovery timed out for %s, "
                           "trying %s", self.name, self.message_url)

        init = self._send_request("initialize", {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "agentic-core", "version": "1.0"},
        })
        self._send_notification("notifications/initialized", {})
        return init

    def reconnect(self):
        """Re-establish the SSE stream after a connection loss."""
        self.close()
        return self.connect()

    def _sse_reader(self):
        event_type = None
        try:
            for line in self._sse_resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if not data:
                        continue
                    if self.message_url is None and (
                            event_type == "endpoint" or data.startswith("/")
                            or data.startswith("http")):
                        if data.startswith("/"):
                            from urllib.parse import urlparse, urlunparse
                            parsed = urlparse(self.url)
                            self.message_url = urlunparse(
                                (parsed.scheme, parsed.netloc, data,
                                 "", "", ""))
                        else:
                            self.message_url = data
                        self._endpoint_event.set()
                        event_type = None
                        continue
                    try:
                        parsed = json.loads(data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        parsed = None
                    if parsed is not None and \
                            not self._handle_server_message(parsed):
                        self._response_queue.put(parsed)
                    event_type = None
        except Exception:
            pass

    def list_tools(self):
        result = self._send_request("tools/list", {})
        return result.get("tools", []) if isinstance(result, dict) else []

    def call_tool(self, tool_name, arguments):
        return self._send_request("tools/call",
                                  {"name": tool_name, "arguments": arguments})

    def _send_request(self, method, params):
        rid = uuid.uuid4().hex
        payload = {"jsonrpc": "2.0", "id": rid, "method": method,
                   "params": params}
        try:
            resp = self._client.post(self.message_url, json=payload)
            resp.raise_for_status()
        except Exception as e:
            raise MCPConnectionError(f"MCP request failed: {e}")
        try:
            data = resp.json()
            if "result" in data:
                return data["result"]
            if "error" in data:
                e = data["error"]
                raise MCPError(
                    f"MCP error {e.get('code')}: {e.get('message')}")
        except (json.JSONDecodeError, ValueError):
            pass
        import queue as _q
        try:
            result = self._response_queue.get(timeout=self.timeout)
        except _q.Empty:
            raise MCPConnectionError(
                f"MCP server '{self.name}' did not respond within "
                f"{self.timeout}s")
        if "result" in result:
            return result["result"]
        if "error" in result:
            e = result["error"]
            raise MCPError(
                f"MCP error {e.get('code')}: {e.get('message')}")
        return result

    def _send_notification(self, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._client.post(self.message_url, json=payload)
        except Exception:
            pass

    def _handle_server_message(self, data):
        """Gracefully handle server-initiated requests/notifications.

        No server→client features are implemented (sampling, roots,
        elicitation): pings are answered, any other server request gets a
        method-not-found error so the server can degrade gracefully, and
        notifications are ignored. Returns True when the message was
        server-initiated and must not be treated as a pending response.
        """
        method = data.get("method")
        if not method:
            return False
        req_id = data.get("id")
        if req_id is None:
            return True  # notification — nothing to answer
        if method == "ping":
            response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            response = {"jsonrpc": "2.0", "id": req_id, "error": {
                "code": -32601,
                "message": f"Method '{method}' is not supported by the "
                           f"MCP client",
            }}
        if self.message_url:
            try:
                self._client.post(self.message_url, json=response)
            except Exception:
                pass
        return True

    def close(self):
        if self._client:
            self._client.close()


class StreamableHTTPTransport:
    """MCP over direct JSON-RPC POST (single URL)."""

    def __init__(self, url, name, timeout=120, auth_token=None, headers=None):
        _require_httpx()
        self.url = url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.auth_token = auth_token
        self.extra_headers = headers or {}
        self._client = None

    def connect(self):
        _require_httpx()
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        h.update(self.extra_headers)
        self._client = httpx.Client(headers=h, timeout=self.timeout)
        init = self._initialize()
        self._send_notification("notifications/initialized", {})
        return init

    def reconnect(self):
        """Re-initialize the HTTP session (new Mcp-Session-Id is captured)."""
        self.close()
        return self.connect()

    def _initialize(self):
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex,
                   "method": "initialize",
                   "params": {
                       "protocolVersion": "2025-03-26",
                       "capabilities": {},
                       "clientInfo": {"name": "agentic-core",
                                      "version": "1.0"}}}
        try:
            resp = self._client.post(self.url, json=payload)
        except httpx.HTTPError as e:
            raise MCPConnectionError(f"MCP request failed: {e}")
        if not resp.is_success:
            raise MCPConnectionError(
                f"HTTP {resp.status_code} for {resp.url}: {resp.text[:300]}")
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._client.headers["Mcp-Session-Id"] = sid
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(resp.text)
        data = resp.json()
        if "result" in data:
            return data["result"]
        if "error" in data:
            e = data["error"]
            raise MCPError(
                f"MCP error {e.get('code')}: {e.get('message')}")
        return data

    def list_tools(self):
        result = self._send_request("tools/list", {})
        return result.get("tools", []) if isinstance(result, dict) else []

    def call_tool(self, tool_name, arguments):
        return self._send_request("tools/call",
                                  {"name": tool_name,
                                   "arguments": arguments})

    def _send_request(self, method, params):
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex,
                   "method": method, "params": params}
        try:
            resp = self._client.post(self.url, json=payload)
        except httpx.HTTPError as e:
            raise MCPConnectionError(f"MCP request failed: {e}")
        if not resp.is_success:
            raise MCPConnectionError(
                f"HTTP {resp.status_code} for {resp.url}: {resp.text[:300]}")
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(resp.text)
        data = resp.json()
        if "result" in data:
            return data["result"]
        if "error" in data:
            e = data["error"]
            raise MCPError(
                f"MCP error {e.get('code')}: {e.get('message')}")
        return data

    @staticmethod
    def _parse_sse(text):
        result = None
        for line in text.split("\n"):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "method" in parsed:
                        # Server-initiated request/notification in the
                        # stream — no server→client features; skip it
                        # and keep looking for our response.
                        continue
                    if "result" in parsed:
                        result = parsed["result"]
                    elif "error" in parsed:
                        e = parsed["error"]
                        raise MCPError(
                            f"MCP error {e.get('code')}: "
                            f"{e.get('message')}")
        return result or {}

    def _send_notification(self, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._client.post(self.url, json=payload)
        except Exception:
            pass

    def close(self):
        if self._client:
            self._client.close()


class StdioTransport:
    """MCP over a subprocess (newline-delimited JSON-RPC on stdio)."""

    def __init__(self, command, args=None, env=None, name="", timeout=120):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.name = name
        self.timeout = timeout
        self._process = None
        self._lock = threading.Lock()
        self._request_id = 0

    def _resolve_command(self):
        """Resolve the command via PATH (handles npx → npx.cmd on Windows)."""
        if os.path.sep in self.command or os.path.isabs(self.command):
            return self.command
        return shutil.which(self.command) or self.command

    def connect(self):
        env = os.environ.copy()
        env.update(self.env)
        env["MCP_SERVER"] = self.name
        resolved = self._resolve_command()
        cmd = [resolved] + self.args
        # No shell=True: with list argv it makes quoting fragile on
        # Windows. .cmd/.bat shims (npx etc.) run through an explicit
        # `cmd /c`, everything else via CreateProcess with proper
        # list2cmdline quoting.
        if os.name == "nt" and os.path.splitext(resolved)[1].lower() in (
                ".cmd", ".bat"):
            cmd = ["cmd", "/c"] + cmd
        try:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, shell=False)
        except OSError as e:
            raise MCPConnectionError(
                f"failed to start MCP server '{self.name}': {resolved}: {e}")
        init = self._send_request("initialize", {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "agentic-core", "version": "1.0"},
        })
        self._send_notification("notifications/initialized", {})
        return init

    def reconnect(self):
        """Restart the server process and re-initialize the session."""
        self.close()
        return self.connect()

    def _alive(self):
        return self._process and self._process.poll() is None

    def list_tools(self):
        result = self._send_request("tools/list", {})
        return result.get("tools", []) if isinstance(result, dict) else []

    def call_tool(self, tool_name, arguments):
        return self._send_request("tools/call",
                                  {"name": tool_name,
                                   "arguments": arguments})

    def _send_request(self, method, params):
        with self._lock:
            self._request_id += 1
            rid = self._request_id
            payload = {"jsonrpc": "2.0", "id": rid, "method": method,
                       "params": params}
            return self._send_and_receive(payload)

    def _send_and_receive(self, payload):
        if not self._alive():
            raise MCPConnectionError(
                f"MCP server '{self.name}' is not running")
        self._write_line(json.dumps(payload))
        # Read until the response to OUR request arrives. Server-initiated
        # requests/notifications (sampling, roots, pings, log messages)
        # are answered or ignored inline — no server→client features.
        for _ in range(64):
            line = self._process.stdout.readline()
            if not line:
                stderr = ""
                try:
                    stderr = self._process.stderr.read(2000).decode(
                        "utf-8", errors="replace")
                except Exception:
                    pass
                raise MCPConnectionError(
                    f"MCP server '{self.name}' closed the connection. "
                    f"stderr: {stderr[:500]}")
            try:
                data = json.loads(
                    line.decode("utf-8") if isinstance(line, bytes) else line)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise MCPConnectionError(
                    f"MCP server '{self.name}' returned invalid JSON: {e}")
            if self._handle_server_message(data):
                continue
            if "result" in data:
                return data["result"]
            if "error" in data:
                e = data["error"]
                raise MCPError(
                    f"MCP error {e.get('code')}: {e.get('message')}")
            return data
        raise MCPConnectionError(
            f"MCP server '{self.name}' sent too many unsolicited messages")

    def _write_line(self, message):
        if self._process and self._process.stdin:
            self._process.stdin.write((message + "\n").encode("utf-8"))
            self._process.stdin.flush()

    def close(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._process = None

    def _handle_server_message(self, data):
        """Gracefully handle server-initiated requests/notifications.

        No server→client features are implemented (sampling, roots,
        elicitation): pings are answered, any other server request gets a
        method-not-found error so the server can degrade gracefully, and
        notifications are ignored. Returns True when the message was
        server-initiated and must not be treated as the pending response.
        """
        method = data.get("method")
        if not method:
            return False
        req_id = data.get("id")
        if req_id is None:
            return True  # notification — nothing to answer
        if method == "ping":
            self._write_response(req_id, result={})
        else:
            self._write_response(req_id, error={
                "code": -32601,
                "message": f"Method '{method}' is not supported by the "
                           f"MCP client",
            })
        return True

    def _write_response(self, req_id, result=None, error=None):
        payload = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result if result is not None else {}
        try:
            self._write_line(json.dumps(payload))
        except Exception:
            pass
