"""MCP transports: SSE, Streamable HTTP, stdio.

httpx is optional (the http_request-tool precedent): stdio works
without it; the HTTP transports raise a clear install hint.
"""

from __future__ import annotations

import json
import logging
import os
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
        resp = self._client.get(sse_url,
                                headers={"Accept": "text/event-stream"})
        resp.raise_for_status()
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
                        self._response_queue.put(json.loads(data))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
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
        resp = self._client.post(self.message_url, json=payload)
        resp.raise_for_status()
        try:
            data = resp.json()
            if "result" in data:
                return data["result"]
            if "error" in data:
                e = data["error"]
                raise RuntimeError(
                    f"MCP error {e.get('code')}: {e.get('message')}")
        except (json.JSONDecodeError, ValueError):
            pass
        import queue as _q
        try:
            result = self._response_queue.get(timeout=self.timeout)
        except _q.Empty:
            raise RuntimeError(f"MCP {self.name}: response timed out "
                               f"after {self.timeout}s")
        if "result" in result:
            return result["result"]
        if "error" in result:
            e = result["error"]
            raise RuntimeError(
                f"MCP error {e.get('code')}: {e.get('message')}")
        return result

    def _send_notification(self, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._client.post(self.message_url, json=payload)
        except Exception:
            pass

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

    def _initialize(self):
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex,
                   "method": "initialize",
                   "params": {
                       "protocolVersion": "2025-03-26",
                       "capabilities": {},
                       "clientInfo": {"name": "agentic-core",
                                      "version": "1.0"}}}
        resp = self._client.post(self.url, json=payload)
        if not resp.is_success:
            raise RuntimeError(f"HTTP {resp.status_code} for {resp.url}: "
                               f"{resp.text[:300]}")
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
            raise RuntimeError(
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
        resp = self._client.post(self.url, json=payload)
        if not resp.is_success:
            raise RuntimeError(f"HTTP {resp.status_code} for {resp.url}: "
                               f"{resp.text[:300]}")
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(resp.text)
        data = resp.json()
        if "result" in data:
            return data["result"]
        if "error" in data:
            e = data["error"]
            raise RuntimeError(
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
                        if "result" in parsed:
                            result = parsed["result"]
                        elif "error" in parsed:
                            e = parsed["error"]
                            raise RuntimeError(
                                f"MCP error {e.get('code')}: "
                                f"{e.get('message')}")
                    except json.JSONDecodeError:
                        pass
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

    def connect(self):
        env = os.environ.copy()
        env.update(self.env)
        env["MCP_SERVER"] = self.name
        cmd = [self.command] + self.args
        # shell=True on Windows so npx & friends resolve from PATH
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, shell=os.name == "nt")
        init = self._send_request("initialize", {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "agentic-core", "version": "1.0"},
        })
        self._send_notification("notifications/initialized", {})
        return init

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
            raise RuntimeError(
                f"MCP server '{self.name}' is not running")
        self._write_line(json.dumps(payload))
        line = self._process.stdout.readline()
        if not line:
            stderr = ""
            try:
                stderr = self._process.stderr.read(2000).decode(
                    "utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"MCP server '{self.name}' closed the connection. "
                f"stderr: {stderr[:500]}")
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(
                f"MCP server '{self.name}' returned invalid JSON: {e}")
        if "result" in data:
            return data["result"]
        if "error" in data:
            e = data["error"]
            raise RuntimeError(
                f"MCP error {e.get('code')}: {e.get('message')}")
        return data

    def _send_notification(self, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._write_line(json.dumps(payload))
        except Exception:
            pass

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
