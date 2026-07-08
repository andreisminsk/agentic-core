"""Web tools: web_search, web_fetch, http_request, time_now.

Adapted from ollama-chat-agentic. Uses stdlib where possible.
Optional dependencies: ddgs (web_search), httpx (http_request), beautifulsoup4 (web_fetch).
"""

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser

# ── web_search ──────────────────────────────────────────────────────────

try:
    from ddgs import DDGS as _DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False


class WebSearchTool:
    name = "web_search"
    description = "Search the web using DuckDuckGo."
    system_prompt = (
        "## web_search\n"
        "Searches the web using DuckDuckGo. Returns titles, URLs, and snippets.\n"
        "Parameters: query (string, required), num_results (int, optional, default 5, max 10)."
    )
    auto_approve = False  # makes external network requests

    def execute(self, params, workdir=None):
        query = params.get("query")
        if not query:
            return "Error: 'query' parameter is required"
        num_results = min(int(params.get("num_results", 5)), 10)
        if _HAS_DDGS:
            return self._search_ddgs(query, num_results)
        return self._search_urllib(query, num_results)

    def _search_ddgs(self, query, num_results):
        try:
            with _DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            if not results:
                return f"[No results for: {query}]"
            parts = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                parts.append(f"{i}. {r.get('title', '')}\n   URL: {r.get('href', '')}\n   {r.get('body', '')}\n")
            return "\n".join(parts)
        except Exception as e:
            return f"[Search error: {e}]"

    def _search_urllib(self, query, num_results):
        try:
            url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            if "anomaly-modal" in html or "bots use DuckDuckGo" in html:
                return ("[Search blocked by CAPTCHA. Install 'ddgs' for reliable search: pip install ddgs]")
            results = _parse_ddg_lite(html, num_results)
            if not results:
                return f"[No results for: {query}]"
            parts = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                parts.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n")
            return "\n".join(parts)
        except Exception as e:
            return f"[Search error: {e}]"


def _parse_ddg_lite(html, max_results):
    results = []
    link_pattern = re.compile(
        r"<a\s[^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>"
        r"|<a\s[^>]*href=['\"]([^'\"]*)['\"][^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>",
        re.DOTALL,
    )
    snippet_pattern = re.compile(r"<td[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>", re.DOTALL)
    snippets = snippet_pattern.findall(html)
    for i, match in enumerate(link_pattern.finditer(html)):
        if i >= max_results:
            break
        title_html = match.group(3) or match.group(1) or ""
        url_raw = match.group(2) or ""
        if not url_raw:
            href_match = re.search(r"href=['\"]([^'\"]*)['\"]", match.group(0))
            url_raw = href_match.group(1) if href_match else ""
        title = re.sub(r'<[^>]+>', '', title_html).strip()
        snippet = snippets[i].strip() if i < len(snippets) else ""
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        uddg_match = re.search(r'uddg=([^&]+)', url_raw)
        if uddg_match:
            url = urllib.parse.unquote(uddg_match.group(1))
        elif url_raw.startswith("//"):
            url = "https:" + url_raw
        else:
            url = url_raw
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


# ── web_fetch ───────────────────────────────────────────────────────────

class WebFetchTool:
    name = "web_fetch"
    description = "Fetch text content from a URL, with optional LLM summarization."
    system_prompt = (
        "## web_fetch\n"
        "Fetches the text content of a web page. Content is converted to clean plain text.\n"
        "Parameters: url (string, required), max_length (int, optional, default 3000), "
        "summarize (bool, optional, default false)."
    )
    auto_approve = False  # makes external network requests

    def __init__(self, llm_client=None, llm_model=None):
        """Optional LLM client for summarization. If None, summarization is skipped."""
        self.llm_client = llm_client
        self.llm_model = llm_model

    def execute(self, params, workdir=None):
        url = params.get("url")
        if not url:
            return "Error: 'url' parameter is required"
        max_length = int(params.get("max_length", 3000))
        summarize = params.get("summarize", False)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if not any(t in content_type.lower() for t in ("text/", "json", "xml", "html")):
                    return f"[Error: Unsupported content type: {content_type}]"
                read_limit = max_length * 5 if summarize else max_length + 1
                data = resp.read(read_limit).decode("utf-8", errors="replace")
                if "html" in content_type.lower():
                    data = _html_to_text(data)
                if summarize and self.llm_client:
                    data = self._summarize(data, url, max_length)
                if len(data) > max_length:
                    data = data[:max_length] + "\n[... truncated]"
                return data
        except urllib.error.HTTPError as e:
            return f"[HTTP Error {e.code}: {e.reason}]"
        except urllib.error.URLError as e:
            return f"[URL Error: {e.reason}]"
        except Exception as e:
            return f"[Error: {e}]"

    def _summarize(self, text, url, max_length):
        prompt = (
            f"Summarize the following web page into a concise, information-dense summary "
            f"of at most {max_length} characters. Preserve key facts, numbers, specs. "
            f"Remove boilerplate and navigation.\n\nURL: {url}\n\nCONTENT:\n{text}\n\nSUMMARY:"
        )
        try:
            resp = self.llm_client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.3},
            )
            if isinstance(resp, dict):
                summary = resp.get("message", {}).get("content", "")
            else:
                summary = resp.message.content
            return summary[:max_length] if summary else text
        except Exception:
            return text


def _html_to_text(html):
    """Convert HTML to clean plain text. Uses BeautifulSoup if available."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside',
                         'noscript', 'iframe', 'form', 'svg']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    except ImportError:
        return _html_to_text_stdlib(html)


def _html_to_text_stdlib(html):
    SKIP_TAGS = frozenset([
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        'noscript', 'iframe', 'form', 'svg',
    ])
    BLOCK_TAGS = frozenset([
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'tr', 'br', 'hr', 'blockquote', 'section',
        'article', 'main', 'figure', 'figcaption',
        'details', 'summary', 'dd', 'dt', 'td', 'th',
    ])

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag in SKIP_TAGS:
                self.skip_depth += 1
            if tag in BLOCK_TAGS and self.skip_depth == 0:
                self.parts.append('\n')

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in SKIP_TAGS:
                self.skip_depth = max(0, self.skip_depth - 1)
            if tag in BLOCK_TAGS and self.skip_depth == 0:
                self.parts.append('\n')

        def handle_data(self, data):
            if self.skip_depth > 0:
                return
            self.parts.append(data)

    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = unescape(''.join(parser.parts))
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n[ \t]*\n[ \t]*', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── http_request ────────────────────────────────────────────────────────

class HttpRequestTool:
    name = "http_request"
    description = "Make HTTP requests (GET/POST/PUT/DELETE) and return structured responses."
    system_prompt = (
        "## http_request\n"
        "Make HTTP requests and return structured responses.\n"
        "Parameters: url (string, required), method (string, optional, default GET), "
        "headers (object, optional), body (string, optional), json_body (object, optional), "
        "timeout (int, optional, default 30), max_length (int, optional, default 5000)."
    )
    auto_approve = False  # makes external network requests

    def execute(self, params, workdir=None):
        url = params.get("url", "")
        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body", "")
        json_body = params.get("json_body")
        timeout = int(params.get("timeout", 30))
        max_length = int(params.get("max_length", 5000))
        if not url:
            return "Error: 'url' is required"
        try:
            import httpx
        except ImportError:
            return "Error: httpx not installed. Install with: pip install httpx"
        default_ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
        req_headers = headers or {}
        if "User-Agent" not in {k.title() for k in req_headers}:
            req_headers.setdefault("User-Agent", default_ua)
        request_kwargs = {"timeout": timeout, "headers": req_headers}
        if json_body is not None:
            request_kwargs["json"] = json_body
        elif body:
            request_kwargs["content"] = body
        try:
            with httpx.Client() as client:
                response = client.request(method, url, **request_kwargs)
        except httpx.TimeoutException:
            return f"Error: Request timed out after {timeout}s"
        except httpx.ConnectError as e:
            return f"Error: Connection failed: {e}"
        except Exception as e:
            return f"Error: {e}"
        parts = [f"Status: {response.status_code} {response.reason_phrase}"]
        for key in ("content-type", "content-length", "location"):
            for k, v in response.headers.items():
                if k.lower() == key:
                    parts.append(f"  {k}: {v}")
        body_text = response.text
        content_type = response.headers.get("content-type", "")
        if "json" in content_type or body_text.strip().startswith(("{", "[")):
            try:
                parsed = json.loads(body_text)
                body_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass
        if len(body_text) > max_length:
            body_text = body_text[:max_length] + f"\n... [truncated, {len(response.text)} chars total]"
        parts.append(f"Body:\n{body_text}")
        return "\n".join(parts)


# ── time_now ────────────────────────────────────────────────────────────

def _tz_offset_str():
    offset = datetime.now().astimezone().utcoffset()
    if offset is None:
        return "+0000"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    h, m = divmod(total // 60, 60)
    return f"{sign}{h:02d}{m:02d}"


def _tz_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation",
            )
            tz_key_name, _ = winreg.QueryValueEx(key, "TimeZoneKeyName")
            winreg.CloseKey(key)
            if tz_key_name:
                return tz_key_name
        except Exception:
            pass
    try:
        import time as _time_mod
        return _time_mod.tzname[0]
    except Exception:
        return "UTC" + _tz_offset_str()


class TimeNowTool:
    name = "time_now"
    description = "Return the current date, time, and timezone."
    system_prompt = (
        "## time_now\n"
        "Return the current date, time, and timezone.\n"
        "Parameters: utc (bool, optional, default false) — also show UTC time."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        show_utc = params.get("utc", False)
        now = datetime.now()
        tz_name = _tz_name()
        tz_offset = _tz_offset_str()
        lines = [
            f"Date: {now.strftime('%Y-%m-%d')}",
            f"Time: {now.strftime('%H:%M:%S')}",
            f"Timezone: {tz_name} (UTC{tz_offset})",
        ]
        if show_utc:
            utc_now = datetime.now(timezone.utc)
            lines.append(f"UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)
