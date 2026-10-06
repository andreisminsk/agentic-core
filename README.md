# agentic_core

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 157](https://img.shields.io/badge/tests-157%20passed-brightgreen.svg)](#testing)
[![Dependencies: 0](https://img.shields.io/badge/dependencies-0%20hard-blue.svg)](#dependencies)

A minimalistic, reusable Python package that gives any LLM chat
application agentic capabilities — file reading/writing/editing, shell
command execution, and extensible tool/skill workflows — through a
text-based block protocol. Supports both Ollama and OpenAI-compatible
APIs via a pluggable backend abstraction with optional TPM rate limiting.

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Backends](#backends)
- [How It Works](#how-it-works)
- [The Block Protocol](#the-block-protocol)
- [Integration Guide](#integration-guide)
- [Architecture Patterns](#architecture-patterns)
- [Integrating With an Existing Feedback Loop](#integrating-with-an-existing-feedback-loop)
- [API Reference](#api-reference)
- [Built-in Tools](#built-in-tools)
- [Graceful Interrupt Handling](#graceful-interrupt-handling-ctrlc)
- [Safety](#safety)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [License](#license)

---

## Overview

`agentic_core` implements a **block-based action protocol** that sits
between an LLM and your application. The model emits structured blocks
(`**WRITE:**`, `**EDIT:**`, `**FILE:**`, `**RUN:**`, `**TOOL:**`,
`**SKILL:**` with `**EOF:**` markers) in its responses. The core parses
these blocks, executes them, and feeds the results (observations) back
into the conversation — a feedback loop that continues until the model
produces a final text-only response.

This pattern was extracted from the
[`ollama-chat-agentic`](https://github.com/) project and distilled into
a standalone library with zero hard dependencies (Python stdlib only).
The LLM client is injected via a backend abstraction — the core never
imports `ollama` or `openai` directly. Two backends ship built-in:
`OllamaBackend` (zero-overhead passthrough) and `OpenAIBackend` (with
optional TPM limiting, history trimming, and retry).

### Why Use It

- **Embed agentic behavior** in any Ollama chat app with ~3 lines of code
- **Zero hard dependencies** — pure Python stdlib, cross-platform
- **Extensible** — add custom tools and skills by subclassing
- **Safe** — built-in command safety checks, blocked/warning lists,
  pluggable confirmation callbacks
- **Context-aware** — automatic truncation prevents context window bloat
- **Battle-tested parser** — handles fence depth, bare EOF, smart quotes,
  markdown blocks, and edge cases from real LLM output
- **Multi-backend** — Ollama and OpenAI-compatible APIs via pluggable
  backends, with optional TPM rate limiting and context trimming
- **Comprehensive test suite** — 157 tests covering parser, executor,
  session loop, streaming, interruption, and backend middleware

---

## Dependencies

**Required:** Python 3.9+ (uses `ast.Constant`, `math.dist`)

**Optional** (installed on-demand by specific features):

| Package | Used by | Install |
|---------|---------|---------|
| `ollama` | `image-analysis` tool, `OllamaBackend` (if constructing inside) | `pip install ollama` |
| `openai` | `OpenAIBackend` | `pip install openai` |
| `tiktoken` | `TokenCounter` (accurate token counting; falls back to char/4) | `pip install tiktoken` |
| `ddgs` | `web_search` tool (reliable mode) | `pip install ddgs` |
| `httpx` | `http_request` tool | `pip install httpx` |
| `beautifulsoup4` | `web_fetch` tool (cleaner HTML parsing) | `pip install beautifulsoup4` |

Without optional packages, the core and all file/shell tools work fully.
`web_search` falls back to a stdlib urllib scraper, and `web_fetch` falls
back to a stdlib `html.parser`-based extractor.

---

## Quick Start

### Installation

No install needed — just copy the `agentic_core/` package into your
project, or add it to your `PYTHONPATH`.

For pip-based projects, you can also install from source:

```bash
git clone https://github.com/yourusername/agentic-core.git
cd agentic-core
pip install -e .  # if a setup.py/pyproject.toml is added
```

### Minimal Usage

```python
from agentic_core import AgenticSession
from ollama import Client

client = Client(host="http://localhost:11434")
session = AgenticSession(client, model="glm-5.1:cloud", workdir=".")

response = session.run("Create a hello.py file that prints hello world")
print(response)
```

### With Streaming

```python
session = AgenticSession(
    client,
    model="glm-5.1:cloud",
    workdir=".",
    on_chunk=lambda chunk: print(chunk, end="", flush=True),
)
response = session.run("List all Python files in this directory")
```

### OpenAI-Compatible Backend (no TPM)

Use `OpenAIBackend` to connect to any OpenAI-compatible endpoint (OpenAI,
Ollama `/v1`, vLLM, LM Studio, etc.):

```python
from agentic_core import AgenticSession
from agentic_core.backends import OpenAIBackend

backend = OpenAIBackend(
    base_url="http://localhost:11434",  # /v1 appended automatically
    api_key="ollama",
    model="glm-5.1:cloud",
    ctx_size=32768,
)
session = AgenticSession(backend=backend, workdir=".")
response = session.run("Create a hello.py file that prints hello world")
```

### OpenAI-Compatible Backend with TPM Limiting

When `tpm_limit` is set, the backend auto-configures a coordinated bundle:
capped history trimming (smaller context = fewer tokens/request), proactive
TPM waiting (rolling 60s window), and aggressive 429-aware retry.

```python
from agentic_core import AgenticSession
from agentic_core.backends import OpenAIBackend

backend = OpenAIBackend(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-5.1",
    ctx_size=32768,
    tpm_limit=50000,       # activates: capped trimming + TPM tracker + aggressive retry
    max_context=16384,      # trim history to 16K tokens (default: 16384)
)
session = AgenticSession(backend=backend, workdir=".")
response = session.run("Create a hello.py file that prints hello world")
```

### With Interactive Confirmation

By default (`auto_approve_safe=False`), the built-in confirm callback
prompts interactively in a TTY with `y/N/auto/all/always/d` options
(see [Confirmation Options](#confirmation-options) below). No custom
callback needed:

```python
session = AgenticSession(
    client,
    model="glm-5.1:cloud",
    workdir=".",
    # auto_approve_safe=False is the default — prompts for WRITE/EDIT/RUN
)
```

To auto-approve everything (autonomous/headless mode):

```python
session = AgenticSession(
    client,
    model="glm-5.1:cloud",
    workdir=".",
    auto_approve_safe=True,
    confirm_callback=lambda atype, details: True,
)
```

To use a custom confirmation policy:

```python
def confirm(action_type, details):
    if action_type == "run":
        return input(f"Run: {details['command']}? [y/N] ").lower().startswith("y")
    if action_type in ("write", "edit"):
        return input(f"{action_type}: {details['path']}? [y/N] ").lower().startswith("y")
    return True

session = AgenticSession(
    client,
    model="glm-5.1:cloud",
    workdir=".",
    confirm_callback=confirm,
)
```

---

## Backends

`agentic_core` uses a **backend abstraction** to support multiple LLM APIs.
The session calls `backend.call(messages, stream, on_chunk)` — the backend
handles transport, rate limiting, and context management.

### Backend ABC

```python
class Backend(ABC):
    def call(self, messages, stream=True, on_chunk=None) -> (text, token_count):
        # 1. TPMTracker.wait_if_needed()  — proactive throttle (if configured)
        # 2. HistoryTrimmer.trim()        — context management (if configured)
        # 3. RetryHandler.execute(_call) — retry on transient errors (if configured)
        # 4. _call(messages, stream)     — backend-specific transport
        # 5. TPMTracker.record_usage()   — record actual usage (if configured)

    @abstractmethod
    def _call(self, messages, stream=True, on_chunk=None) -> (text, token_count):
        """Pure transport. Calls on_chunk(text) per streaming piece.
        Does NOT catch KeyboardInterrupt — propagates to session."""
```

All middleware components are `Optional` and default to `None`. For
`OllamaBackend` (all `None`), `call()` is a direct passthrough to `_call()`
with zero overhead.

### OllamaBackend

Thin wrapper around a duck-typed client. **Zero middleware** — all
components are `None`, preserving the exact behavior of direct
`client.chat()` calls. The client is injected (not imported).

```python
from agentic_core.backends import OllamaBackend
from ollama import Client

backend = OllamaBackend(
    client=Client(host="http://localhost:11434"),
    model="glm-5.1:cloud",
    temperature=0.0,
)
session = AgenticSession(backend=backend, workdir=".")
```

**Backward compatibility:** `AgenticSession(client=..., model=...)` auto-wraps
the client in `OllamaBackend`. Existing code works unchanged.

### OpenAIBackend

Uses `openai.Client` with `/v1/chat/completions`. Works with any
OpenAI-compatible endpoint: OpenAI, Ollama `/v1`, vLLM, LM Studio, etc.

**Auto-configures middleware based on `tpm_limit`:**

| | Without `tpm_limit` | With `tpm_limit` |
|---|---|---|
| **TokenCounter** | Always on | Always on |
| **HistoryTrimmer** | Full `ctx_size` budget | Capped at `max_context` (default 16384) |
| **RetryHandler** | Light: connection/500 only, 5s backoff | Aggressive: 429/rate_limit/connection, 20s backoff |
| **TPMTracker** | Off | Rolling 60s window, proactive wait |

The coupling is intentional: enabling TPM means "I'm on a rate-limited
endpoint, so trim aggressively, wait proactively, and retry reactively."

#### `ctx_size` vs `max_context`

These two parameters control how much conversation history is sent to the
model. They interact but serve different purposes:

| Parameter | Role | Used when |
|---|---|---|
| `ctx_size` | The model's full context window (e.g. 128K for gpt-5.1). Upper bound for trimming. Also caps `max_completion_tokens` at `min(ctx_size, 8192)`. | Always |
| `max_context` | Trim budget cap. When `tpm_limit` is set, the trimmer uses `min(ctx_size, max_context)` instead of the full `ctx_size`. Smaller = fewer tokens/request = stretches TPM budget. | Only with `tpm_limit` |

**Without `tpm_limit`:** `max_context` is ignored. The trimmer uses `ctx_size`
as the budget — the model sees the full context window.

**With `tpm_limit`:** The trimmer uses `min(ctx_size, max_context)` (default
16384). A smaller `max_context` means fewer tokens per request (stretching
the TPM budget) at the cost of less conversation history visible to the model.

**What gets trimmed:** The entire message list counts against the budget
(system prompt + all history + latest message). The system prompt is always
preserved. Middle messages are dropped oldest-first (replaced with a summary).
The last message is preserved but may be truncated if still over budget.
`reserve_output=2048` reserves space for the model's response, so the effective
input budget is `trim_budget - 2048`.

```python
from agentic_core.backends import OpenAIBackend

# Without TPM (light retry, full-ctx trimming)
backend = OpenAIBackend(
    base_url="http://localhost:11434",
    api_key="ollama",
    model="glm-5.1:cloud",
    ctx_size=32768,
)

# With TPM (capped trimming + proactive wait + aggressive retry)
backend = OpenAIBackend(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-5.1",
    ctx_size=32768,
    tpm_limit=50000,
    max_context=16384,
)
```

### Middleware Components

Each component can also be used standalone or manually attached to any
backend via the `Backend` base class attributes:

| Component | Purpose | Dependency |
|---|---|---|
| `TokenCounter` | tiktoken-based counting, char/4 fallback | `tiktoken` (optional) |
| `TPMTracker` | Rolling 60s window, proactive throttle before send | stdlib |
| `HistoryTrimmer` | Trim to token budget, preserve tool-call pairs, summarize dropped | `TokenCounter` |
| `RetryHandler` | Jittered exponential backoff, `Retry-After` header parsing | stdlib |

```python
from agentic_core.backends import OllamaBackend, TokenCounter, TPMTracker, RetryHandler, HistoryTrimmer

# Manually attach middleware to Ollama (advanced)
backend = OllamaBackend(client, "glm-5.1:cloud")
backend._token_counter = TokenCounter("glm-5.1:cloud")
backend._tpm_tracker = TPMTracker(tpm_limit=100000, quiet=False)
backend._retry = RetryHandler(triggers=["429", "connection"], max_retries=3)
backend._trimmer = HistoryTrimmer(backend._token_counter, max_tokens=32768)
```

---

## How It Works

### The Feedback Loop

```
User input
    │
    ▼
┌──────────┐         ┌──────────┐
│  Ollama  │ ◄──────► │  Session │  (feedback loop, max 3 rounds)
│  Client  │          │          │
└──────────┘          └────┬─────┘
                           │ parse_actions(text)
                           ▼
                    ┌──────────┐
                    │  Parser  │ → [{type, path, code, params, ...}, ...]
                    └────┬─────┘
                         │ for each action
                         ▼
               ┌──────────────────┐
               │  ActionExecutor   │
               │  WRITE → write    │
               │  EDIT  → replace  │
               │  FILE  → read     │
               │  RUN   → shell    │
               │  TOOL  → registry │
               │  SKILL → registry │
               └────────┬─────────┘
                        │ observations[]
                        ▼
               fed back as user message → next model call
```

1. User text is added to conversation history
2. The model generates a response containing action blocks
3. `parse_actions()` extracts structured actions from the response
4. `ActionExecutor` runs each action (file I/O, shell, tool, skill)
5. Observations are fed back as a user message
6. Steps 2–5 repeat (up to `MAX_FEEDBACK_ROUNDS=3`) until the model
   produces a text-only response with no action blocks
7. The final prose (with blocks stripped) is returned to the caller

### The Block Protocol

The model communicates actions through a simple text format:

| Block | Purpose | Example |
|-------|---------|---------|
| `**WRITE:**` | Create/overwrite a file | `**WRITE:**\`hello.py\`` |
| `**EDIT:**` | Search/replace in existing file | `**EDIT:**\`app.py\`` |
| `**FILE:**` | Read a file into context | `**FILE:**\`README.md\`` |
| `**RUN:**` | Execute a shell command | `**RUN:**` + ` ```bash ` block |
| `**TOOL:**` | Invoke a registered tool | `**TOOL:**\`read_file\`` |
| `**SKILL:**` | Invoke a registered skill | `**SKILL:**\`code_review\`` |
| `**EOF:**` | Close any block | `**EOF:**\`hello.py\`` |

Each block (except RUN) is closed by a matching `**EOF:**` marker with
the same path/name. RUN blocks use fenced code blocks (```bash ... ```).

---

## Integration Guide

### Basic Integration

Drop the `agentic_core/` package into your project and import it:

```python
from agentic_core import AgenticSession

session = AgenticSession(
    client=my_client,       # any object with .chat(model=, messages=, stream=)
    model="glm-5.1:cloud",
    workdir=".",
)
response = session.run(user_input)
```

### Custom System Prompt

```python
from agentic_core import build_system_prompt

prompt = build_system_prompt(
    name="myapp",
    tools=True,
    skills=True,
    extra_sections=["Custom rule: always respond in French."],
)
session = AgenticSession(client, model="glm-5.1:cloud", system_prompt=prompt)
```

### Custom Tools

```python
from agentic_core.tools import Tool, register

class WeatherTool(Tool):
    name = "weather"
    description = "Get current weather."
    system_prompt = (
        "## weather\n"
        "Get weather for a city. Parameters: city (string, required)."
    )

    def execute(self, params, workdir=None):
        city = params.get("city", "")
        # ... fetch weather ...
        return f"Weather in {city}: sunny, 22°C"

register(WeatherTool())
```

The tool's `system_prompt` is automatically injected into the model's
system prompt when `tools=True`.

### Custom Skills

```python
from agentic_core.skills import Skill, register as register_skill

class CodeReviewSkill(Skill):
    name = "code_review"
    description = "Review code for issues."
    system_prompt = "When invoked, review the provided code for bugs and style issues."
    parameters = {
        "path": {"type": "string", "required": True, "description": "File to review"}
    }

    def execute(self, params, workdir=None, session=None):
        return f"[Skill code_review invoked] Reviewing {params.get('path')}..."

register_skill(CodeReviewSkill())
```

### MCP (Model Context Protocol)

`agentic_core.mcp` connects to external MCP tool servers, discovers
their tools, and registers each as a first-class tool named
`mcp_<server>_<tool>`. Three transports: **SSE** (url ending with
`/sse`), **Streamable HTTP** (any other url), and **stdio**
(`command` + `args` subprocess). `httpx` is optional — stdio works
without it; HTTP transports raise a clear install hint.

The manager is host-injected: you provide the config, optionally a
confinement checker and an auth-token resolver.

```python
from agentic_core.mcp import MCPManager, MCPUsageTool

mcp = MCPManager(
    # REQUIRED: callable returning the mcpServers dict — read it
    # from your own config source, fresh on every call
    get_config=lambda: {
        "context7": {
            "url": "https://mcp.context7.com/mcp",
            "auth_token": "...",          # or auth_token_env: "MY_TOKEN"
            "timeout": 120,
            "auto_approve": False,         # per-server confirmation opt-out
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
        },
    },
    # OPTIONAL: (path, workdir) -> error-string-or-None — when set,
    # output-file saves (large text / binary results) are refused
    # outside the allowed roots. Pass None for no confinement.
    path_checker=my_check_path,
    # OPTIONAL: env-file fallback for auth_token_env tokens
    env_file="~/.myapp/.env",
)

# Lifecycle is the HOST's policy — connect when it makes sense:
mcp.connect()                    # all enabled servers, in parallel
mcp.connect("context7")          # one server
mcp.sync()                       # drop removed/disabled; reconnect dead
mcp.register_tools()             # into the agentic-core registry
mcp.close_all()                  # shutdown
```

**Lazy activation (recommended).** MCP tools are registered but
excluded from the system prompt (`tools_system_prompt` skips
`mcp_*` names) — N servers × M tools cost nothing until used.
Register the meta-tool and add one stub line to your prompt:

```python
from agentic_core.tools import register

register(MCPUsageTool(mcp))       # serves full tool docs on demand

# In your system prompt (extra_sections):
# "MCP TOOLS AVAILABLE (not shown here): context7, filesystem.
#  Invoke TOOL:mcp_usage to connect and see their full documentation."
```

When the user asks for MCP, the model calls `mcp_usage` — which
connects on demand, returns the full generated docs as its
observation, and the real tools become callable. One extra
round-trip on first use; zero prompt cost otherwise.

**Enablement gating.** `register_tools(enabled_check=name -> bool)`
lets the host apply its own tool-enablement config to late-registered
MCP tools (they bypass any boot-time filter otherwise).

### Custom Client (Non-Ollama)

The core accepts any client object with a `.chat()` method matching the
Ollama API shape:

```python
class MyClient:
    def chat(self, model, messages, stream, options=None):
        # Return a dict (non-stream) or iterable of dicts (stream)
        # Each dict must have: {"message": {"content": "..."}}
        ...

session = AgenticSession(MyClient(), model="my-model", workdir=".")
```

---

## Architecture Patterns

`agentic_core` is designed to be embedded in a variety of application
architectures. Below are extended descriptions of the patterns in which
it can be used.

### 1. CLI Chat Application

The simplest integration — a terminal-based chat loop. The core handles
the agentic loop; your app handles input/output and session persistence.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Terminal    │────►│  Your App    │────►│ AgenticSession│
│  (stdin/out) │◄────│  (REPL loop) │◄────│ (feedback)   │
└──────────────┘     └──────────────┘     └──────────────┘
```

**Your app provides:**
- Input reading (stdin, readline, prompt toolkit)
- Output display (streaming chunks, colored output)
- Session save/load (serialize `session.history` to JSON)
- Slash commands (`/help`, `/reset`, `/save`)

**The core provides:**
- Model calling and streaming
- Action parsing and execution
- Observation feedback loop
- Tool/skill dispatch

**Examples:** This project includes two runnable examples:

- **`example.py`** — Minimal CLI chat loop. Uses the built-in confirmation
  system (`y/N/auto/all/always/d`), registers a custom `EchoTool` and
  `GreetingSkill`, and handles `AgenticInterrupted` (Ctrl+C) gracefully.
  Start here for the simplest integration.

- **`example-callback.py`** — Custom confirmation callback. Demonstrates a
  selective approval policy: auto-approves safe tools and safe RUN commands,
  shows colored diff previews for EDIT, prompts `[y/N]` for everything else.
  Uses a custom config path and `max_feedback_rounds=5`. Use this as a
  template when you need fine-grained control over what the agent can do.

### 2. Web API Backend

Embed the core in a FastAPI/Flask backend. Each user session maps to an
`AgenticSession` instance. The API streams model chunks to the frontend
via Server-Sent Events (SSE) or WebSocket.

```
┌────────┐     ┌──────────┐     ┌────────────────┐     ┌──────────────┐
│ Browser│────►│ FastAPI  │────►│ AgenticSession  │────►│ Ollama Server│
│ (SSE)  │◄────│ Endpoint │◄────│ (per-session)  │◄────│ (local/remote)│
└────────┘     └──────────┘     └────────────────┘     └──────────────┘
```

**Your app provides:**
- HTTP endpoints (`POST /chat`, `GET /chat/stream`)
- Session management (dict of session_id → AgenticSession)
- Authentication and rate limiting
- Frontend streaming (SSE/WebSocket)
- File sandboxing (restrict `workdir` per user)

**The core provides:**
- The agentic feedback loop
- Tool/skill execution
- Observation generation

**Key consideration:** Set `workdir` to a per-user sandbox directory to
isolate file operations. Use `confirm_callback` to enforce permission
policies server-side.

```python
from fastapi import FastAPI
from agentic_core import AgenticSession

app = FastAPI()
sessions = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, message: str):
    if session_id not in sessions:
        sessions[session_id] = AgenticSession(
            client, model="glm-5.1:cloud",
            workdir=f"/tmp/sessions/{session_id}",
        )
    response = sessions[session_id].run(message)
    return {"response": response}
```

### 3. Multi-Agent Orchestration

Run multiple `AgenticSession` instances with different models, tools,
and system prompts. A "router" agent delegates tasks to specialist
agents.

```
                    ┌──────────────┐
                    │  Router Agent │
                    │  (light model)│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Coder    │ │ Reviewer │ │ Tester  │
        │ Agent    │ │ Agent    │ │ Agent    │
        │(code tools)│(read tools)│(run tools)│
        └──────────┘ └──────────┘ └──────────┘
```

**Each agent has:**
- Its own `AgenticSession` with a tailored system prompt
- A specific tool subset (e.g. coder gets `write_file`, reviewer gets
  only `read_file`)
- A shared or isolated `workdir`

**The orchestrator:**
- Calls the router agent with the user request
- Parses the router's output to determine which specialist to invoke
- Passes the task to the specialist agent
- Collects and synthesizes results

```python
coder = AgenticSession(client, model="glm-5.1:cloud",
                       workdir="./workspace", tools=True)
reviewer = AgenticSession(client, model="glm-5.1:cloud",
                          workdir="./workspace", tools=True)

# Coder writes code
coder.run("Create a calculator.py module")

# Reviewer reviews it
reviewer.run("Review calculator.py for bugs")
```

### 4. Background Job Worker

Embed the core in a background worker (Celery, RQ, or a simple thread
pool). Long-running agentic tasks are submitted as jobs and processed
asynchronously.

```
┌────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ Client │───►│ Job Queue│───►│ Worker       │───►│ AgenticSession│
│        │◄───│ (Redis)  │◄───│ (agentic)    │◄───│ (feedback)   │
└────────┘    └──────────┘    └──────────────┘    └──────────────┘
```

**Your app provides:**
- Job submission and status tracking
- Worker process management
- Result storage (database, file system)

**The core provides:**
- The full agentic loop within each job

**Key consideration:** Set `stream=False` and `on_chunk=None` for
headless operation. Use `confirm_callback=lambda *a: True` for
fully autonomous execution.

### 5. IDE Plugin / Editor Integration

Embed the core in a VS Code extension, Neovim plugin, or JetBrains
plugin. The agentic session operates on the user's project directory.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Editor UI    │────►│ Plugin Host  │────►│ AgenticSession│
│ (chat panel) │◄────│ (bridge)    │◄────│ (workdir=     │
└──────────────┘     └──────────────┘     │  project root)│
                                          └──────────────┘
```

**Your app provides:**
- Editor-specific UI (chat panel, diff view, inline suggestions)
- Project/workspace detection (set `workdir` to project root)
- File watching and refresh (notify editor after WRITE/EDIT actions)
- Diff display (use `make_unified_diff` from `edit_utils`)

**The core provides:**
- Agentic file operations on the project
- Tool/skill execution

**Key consideration:** Hook into the `ActionExecutor` to emit editor
events after each WRITE/EDIT. Override or subclass to add callbacks.

### 6. CI/CD Pipeline Integration

Use the core as an autonomous code fixer or test generator in a CI/CD
pipeline. Triggered by a failed build or PR, the agentic session
analyzes errors and proposes fixes.

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ CI Trigger│──►│ AgenticSession│──►│ Git Commit  │──►│ PR Update│
│ (build fail)│  │ (autonomous)  │  │ (auto)      │  │          │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

**Your app provides:**
- CI event handling (webhook, pipeline step)
- Error log extraction (feed as user input to the session)
- Git operations (commit fixes, push branch, open PR)
- Safety guardrails (restrict `workdir`, auto-approve all)

**The core provides:**
- Autonomous code analysis and modification
- Test execution via RUN blocks
- Iterative fix-test loop (feedback loop)

### 7. Embedded Sub-Agent (Library Mode)

Use the core as a subroutine within a larger LLM application. Instead
of a persistent session, create an `AgenticSession` for a single task,
get the result, and discard it.

```
┌──────────────────────────────────────┐
│  Your LLM Application                 │
│  ┌────────────────────────────────┐  │
│  │  Main logic (your code)         │  │
│  │  ┌──────────────────────────┐   │  │
│  │  │ AgenticSession (ephemeral)│   │  │
│  │  │ .run("fix this file")    │   │  │
│  │  │ .run("run the tests")    │   │  │
│  │  └──────────────────────────┘   │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

**Use case:** Your app handles conversation flow, but delegates file
operations to the core. Create a session, run a task, extract the result,
optionally inspect `session.history` for details.

```python
def fix_file(path, error_msg):
    session = AgenticSession(client, model="glm-5.1:cloud", workdir=".")
    return session.run(f"The file {path} has this error: {error_msg}. Fix it.")
```

---

## Integrating With an Existing Feedback Loop

If your app already has its own chat loop, `AgenticSession.run()` is a
blocking, self-contained loop — it calls the model, parses actions,
executes them, feeds observations back, and repeats internally. Two loops
need to coexist. Five strategies, from simplest to most flexible:

### Strategy A: Black-box (use `run()` as-is)

The app treats `session.run(user_text)` as a single blocking call. The
agentic loop happens internally; the app's loop just sees one "turn" that
took longer.

```
App loop:  input → session.run() → [internal N rounds] → display → input → ...
```

**Works when:** the app doesn't need to intervene between agentic rounds.
Streaming is visible via `on_chunk`; action confirmation via
`confirm_callback`.

**Fails when:** the app needs to inject context mid-loop, display action
progress between rounds, handle user interrupts, or apply its own
round-level logic (rate limiting, cost tracking, context pruning).

### Strategy B: Step API (expose the loop)

Instead of `run()`, expose `step()` — one round of (model call → parse →
execute → observe). The app's loop calls `step()` until it signals "done"
(no actions found).

```
App loop:  input → while not done: session.step() → display → input → ...
```

**Works when:** the app wants full control over round timing, wants to
interleave its own logic, or needs to pause/resume.

**This is the cleanest answer for existing-loop apps.** The app keeps its
loop; the core provides stateful single-round execution. Loop detection
and round limits move to the app's side (or the core tracks them
internally and `step()` returns a "stop" signal).

### Strategy C: Hooks/callbacks on `run()`

Keep `run()` but add round-level hooks: `on_round_start`,
`on_actions_parsed`, `on_actions_executed`. The app observes but doesn't
control flow.

**Works when:** the app only needs *visibility* (logging, progress display,
cost tracking), not *control*.

**Fails when:** the app needs to skip rounds, inject extra context, or
abort mid-loop.

### Strategy D: Manual mode (parser + executor only)

The app keeps its own loop entirely. It calls the model itself, passes the
response to `parse_actions()`, runs `executor.execute_actions()`, and
manages history manually. `AgenticSession` isn't used at all.

```
App loop:  input → call_model() → parse_actions() → execute_actions() → feed back → ...
```

**Works when:** the app has a sophisticated existing loop (context
management, multi-model routing, custom history compression) and only
wants the protocol parsing + execution.

**Cost:** the app reimplements loop detection, round limits, and history
management. But it gets maximum control — zero loop conflict.

### Strategy E: Hybrid (session as stateful helper)

The app uses `AgenticSession` for history and system prompt management,
but calls the model itself and uses `parse_actions()` +
`executor.execute_actions()` for the agentic parts. The session's `run()`
is never called.

```python
# App owns the loop
response = app_client.chat(...)
actions = parse_actions(response)
observations = session.executor.execute_actions(actions)
session.history.append({"role": "assistant", "content": response})
session.history.append({"role": "user", "content": "\n".join(observations)})
# repeat...
```

**Works when:** the app wants the core's infrastructure (prompt building,
tool/skill registry, executor) but not its loop control.

### Recommendation

**For apps that already have a feedback loop, Strategy B (step API) is
the right answer.** It's a small addition to the current design:

- `run()` becomes a convenience wrapper that calls `step()` in a loop
- `step()` does one round and returns `(response_text, actions, observations, is_done)`
- The app can call `step()` from its own loop, interleaving any logic
- Loop detection and round limits stay inside the session, exposed as `is_done`

This preserves the simple `run()` API for new apps while giving existing
apps a clean integration point. Strategies D and E remain available for
apps that want even more control.

The key insight: **the core should offer the loop as a convenience, not a
constraint.** `run()` is the convenience; `step()` (or manual
parse+execute) is the escape hatch.

---

## API Reference

### AgenticSession

```python
AgenticSession(
    client=None,             # Ollama-compatible client (duck-typed) — auto-wrapped in OllamaBackend
    model=None,              # Model name string (e.g. "glm-5.1:cloud")
    backend=None,            # Explicit Backend instance (overrides client=)
    workdir=".",             # Working directory for file ops
    tools=True,              # Enable TOOL: protocol + built-in tools
    skills=False,            # Enable SKILL: protocol
    system_prompt=None,      # Custom prompt (None = auto-build)
    name="uhu",              # Agent name for default prompt
    confirm_callback=None,  # fn(action_type, details) -> bool
    auto_approve_safe=False, # Auto-approve safe shell commands + WRITE/EDIT
    stream=True,             # Stream model responses
    temperature=None,        # Override default (0.0)
    on_chunk=None,           # Callback(chunk_text) for streaming display
    max_feedback_rounds=None,# Override default (3) agentic loop rounds
    config_filename="agentic-core.json",  # Persistent auto-approval config file
)
```

Pass either `client=` (auto-wrapped in `OllamaBackend`) or `backend=`
(explicit `OllamaBackend` / `OpenAIBackend`). If both are given, `backend=`
takes precedence.

**Methods:**
- `.run(user_text)` → str: Run the feedback loop, return final prose
- `.reset()`: Clear conversation history (keep system prompt), reset loop detection

**Attributes:**
- `.history`: List of `{"role": ..., "content": ...}` messages
- `.executor`: The `ActionExecutor` instance
- `.system_prompt`: The active system prompt string
- `.backend`: The `Backend` instance handling model calls

### Backend Classes

```python
from agentic_core.backends import Backend, OllamaBackend, OpenAIBackend

# OllamaBackend — zero middleware, thin wrapper around duck-typed client
OllamaBackend(client, model, temperature=0.0)

# OpenAIBackend — auto-configures middleware based on tpm_limit
OpenAIBackend(
    base_url,              # e.g. "http://localhost:11434" (/v1 appended automatically)
    api_key,               # API key string
    model,                 # model name
    ctx_size=8192,         # context window size (for trimming budget)
    temperature=0.0,        # model temperature
    thinking=False,        # whether backend supports thinking tokens
    tpm_limit=None,        # TPM limit (activates full middleware bundle)
    max_context=None,      # trim budget cap when tpm_limit is set (default 16384)
    quiet=False,            # suppress TPM/retry status messages
)
```

### Middleware Components

```python
from agentic_core.backends import TokenCounter, TPMTracker, RetryHandler, HistoryTrimmer

TokenCounter(model, per_message_overhead=4)   # tiktoken with char/4 fallback
TPMTracker(tpm_limit, quiet=False)            # rolling 60s window
RetryHandler(triggers=None, initial_wait=20, max_wait=60, max_retries=3, quiet=False)
HistoryTrimmer(token_counter, max_tokens, reserve_output=2048)
```

### Tool Base Class

```python
class MyTool(Tool):
    name = "my_tool"
    description = "What it does."
    system_prompt = "## my_tool\nHow to use it. Parameters: ..."
    auto_approve = True

    def execute(self, params, workdir=None):
        return "observation string"
```

Register with `register(MyTool())`.

### Skill Base Class

```python
class MySkill(Skill):
    name = "my_skill"
    description = "What it does."
    system_prompt = "Instructions for the model..."
    parameters = {"path": {"type": "string", "required": True}}

    def execute(self, params, workdir=None, session=None):
        return "observation string"
```

Register with `register_skill(MySkill())`.

### System Prompt Customization

There are three levels of control over the system prompt:

**Level 1 — Full override** (pass `system_prompt=` to `AgenticSession`):

```python
session = AgenticSession(
    client, model="glm-5.1:cloud",
    system_prompt="You are a French coding assistant...",
)
```

Completely replaces the default. The consumer is responsible for including
all protocol instructions (WRITE/EDIT/FILE/RUN/TOOL/SKILL format).

**Level 2 — Builder with options** (use `build_system_prompt()`):

```python
from agentic_core import build_system_prompt

prompt = build_system_prompt(
    name="myapp",           # agent identity
    tools=True,             # include TOOL: rules
    skills=True,            # include SKILL: rules
    extra_sections=[        # append custom sections
        "Always respond in French.",
        "Never delete files without asking.",
    ],
    identity=None,          # replace base prompt, keep tools/skills rules
)
session = AgenticSession(client, model="glm-5.1:cloud", system_prompt=prompt)
```

**Level 3 — Default** (no args):

`AgenticSession` auto-builds with `name="uhu"`, `tools=True`,
`skills=False`, and appends tool/skill registry prompts automatically.

**What the builder includes automatically:**

- Current date/time with timezone
- Agent name
- Block protocol instructions (WRITE/EDIT/FILE/RUN + EOF)
- Platform shell guidance (Windows/Unix)
- Tool rules (when `tools=True`) + tool registry system prompts
- Skill rules (when `skills=True`) + skill registry system prompts
- Call-and-wait rules (when tools or skills enabled)
- Time awareness guidance (when `tools=True`)

**What `identity=` does:** replaces the base prompt (name, protocol, output
discipline) but keeps the tools/skills/platform sections. Useful for custom
agent personas that still need the protocol.

---

## Built-in Tools

| Tool | Auto-approve | Description | Optional Dependency |
|------|:---:|-------------|:---:|
| `read_file` | ✅ | Read file contents (offset/limit) | — |
| `write_file` | ✅ | Create/overwrite/append file | — |
| `replace_in_file` | ✅ | Exact string replacement | — |
| `list_files` | ✅ | List directory (recursive, glob) | — |
| `find_file` | ✅ | Find files by name pattern | — |
| `search_in_files` | ✅ | Regex search across files | — |
| `peek_file` | ✅ | Head + tail of a file | — |
| `mkdir` | ✅ | Create directory tree | — |
| `copy_file` | ✅ | Copy file or directory | — |
| `move_file` | ✅ | Move/rename file or directory | — |
| `web_search` | ❌ | Search the web via DuckDuckGo | `ddgs` (falls back to urllib) |
| `web_fetch` | ✅ | Fetch URL content, optional LLM summary | `beautifulsoup4` (falls back to stdlib) |
| `http_request` | ❌ | HTTP GET/POST/PUT/DELETE | `httpx` (required) |
| `time_now` | ✅ | Current date/time/timezone | — |
| `calculator` | ✅ | Safe math expression evaluation | — |
| `image-analysis` | ❌ | Analyze images via vision model | `ollama` (required) |

### Configuring Image Analysis

The `image-analysis` tool is registered by default with no arguments,
which means it lazily creates an `ollama.Client(host="http://localhost:11434")`
and uses the model **`gemma4:31b-cloud`**. To use a different model or
share an existing client, re-register with custom arguments:

```python
from agentic_core.tools import register, ImageAnalysisTool

# Option 1: Specify model + base_url (client created lazily)
register(ImageAnalysisTool(model="llama3.2-vision:11b", base_url="http://localhost:11434"))

# Option 2: Inject an existing client (recommended — shares connection)
register(ImageAnalysisTool(client=my_client, model="gemma4:31b-cloud"))

# Option 3: Use the session's client
session = AgenticSession(client, model="glm-5.1:cloud", ...)
register(ImageAnalysisTool(client=session.client, model="gemma4:31b-cloud"))
```

The tool validates image MIME type (JPEG, PNG, GIF, BMP, WebP, TIFF, ICO)
and enforces a 20MB size limit. Images are base64-encoded and sent to the
vision model with the prompt.

---

## Graceful Interrupt Handling (Ctrl+C)

The core catches `KeyboardInterrupt` at three boundaries and raises a
clean `AgenticInterrupted` exception, ensuring conversation history
remains consistent:

| Phase | What happens | History fix |
|-------|-------------|------------|
| **Streaming** | Ctrl+C during model response | Partial text saved as assistant message; if no text, user message removed |
| **Actions** | Ctrl+C during action execution | Observations so far saved as user message |
| **Confirmation** | Ctrl+C at `confirm_callback` prompt | Same as actions — observations saved |

The app catches `AgenticInterrupted` and returns to the user prompt:

```python
from agentic_core import AgenticSession, AgenticInterrupted

while True:
    user_input = input("you> ")
    try:
        response = session.run(user_input)
    except AgenticInterrupted as e:
        print(f"\n[interrupted during {e.phase}]")
        continue  # history is clean, safe to continue
    print(response)
```

**Design principle:** the core owns state consistency (history is never
left half-written); the app owns UX (what "interrupted" looks like to the
user). The core never swallows the interrupt — it always re-raises so the
app can decide whether to continue, retry, or exit.

`AgenticInterrupted` has two attributes:
- `.partial_text`: text collected before the interrupt (partial model
  response or observations from actions executed so far)
- `.phase`: where it occurred — `'streaming'`, `'actions'`, or `'confirmation'`

---

## Safety

The core includes built-in safety mechanisms:

- **Blocked commands**: `rm -rf /`, `mkfs`, `dd`, `shutdown`, `format`
  — never executed
- **Warning commands**: `rm`, `del`, `git push`, `pip uninstall` —
  require confirmation even with auto-approve
- **Safe commands**: `ls`, `cat`, `grep`, `dir`, `type` — auto-approved
  (only when `auto_approve_safe=True`)
- **Shell chaining**: `&&`, `||`, `|`, `;` — always require confirmation
- **File backup**: Overwritten files are cached in `.uhu/.cache/`
- **Context limits**: Observations are truncated to prevent context bloat
- **Loop detection**: Identical actions repeated 3+ times are stopped
- **Max feedback rounds**: Agentic loop stops after `max_feedback_rounds` (default 3)

### Confirmation Options

When the default `confirm_callback` is active (interactive TTY), each
action prompt offers:

| Option | Behavior |
|--------|----------|
| `y`/`yes` | Approve once |
| `n`/`no`/`Enter` | Reject (default) |
| `auto` | Approve this path/command for the rest of the session |
| `all` | Approve ALL actions for the rest of the session |
| `always` | Approve persistently — saved to config file |
| `d`/`diff` | Show colored diff preview (EDIT) before deciding |

Persistent approvals are saved to `agentic-core.json` (configurable via
`config_filename`) in the workdir. Non-interactive contexts (no TTY)
auto-approve everything.

---

## Project Structure

```
agentic_core/
├── __init__.py          # Public API: AgenticSession, backends, parse_actions, build_system_prompt
├── constants.py         # Limits, safety lists, ANSI colors, platform info
├── parser.py            # Block extraction engine (protocol parser)
├── system_prompt.py     # Modular system prompt builder
├── actions.py           # ActionExecutor: dispatch + confirmation + RUN/TOOL/SKILL
├── safety.py            # CommandSafety mixin: blocked/warning/safe classification
├── file_ops.py          # FileOperations mixin: WRITE/EDIT/FILE handlers
├── utils.py             # Shared utilities: agent_print, tool_print, _truncate
├── backends.py          # Backend ABC + OllamaBackend + OpenAIBackend + middleware
├── matching.py          # Layered fuzzy/exact matching for EDIT
├── edit_utils.py        # Diff/summary helpers
├── exceptions.py        # AgenticInterrupted
├── session.py           # Feedback loop orchestrator
├── tools/
│   ├── __init__.py      # Tool base class + registry + auto-registration
│   ├── fs.py            # 10 built-in file system tools
│   ├── web.py           # web_search, web_fetch, http_request, time_now
│   ├── calculator.py    # Safe AST-based math expression evaluator
│   └── image_analysis.py # Vision model image analysis
└── skills/
    ├── __init__.py      # Skill registry + system prompt builder
    └── base.py          # Skill + PromptOnlySkill base classes
```

### Architecture: Mixin Composition

`ActionExecutor` uses mixin composition to keep concerns separated while
maintaining a single public API:

```
ActionExecutor(CommandSafety, FileOperations)
├── CommandSafety  (safety.py)    → _check_command_safety, _get_base_command, _is_safe_command
├── FileOperations (file_ops.py) → _do_write, _do_edit, _do_file, _preview_edit_diff
└── ActionExecutor (actions.py)  → execute_actions, _should_approve, _do_run, _do_tool, _do_skill
```

### Architecture: Backend Abstraction

`AgenticSession` delegates model calls to a `Backend`, which orchestrates
optional middleware (TPM, trimming, retry) around the transport layer:

```
AgenticSession
  └── backend.call(messages, stream, on_chunk)
        ├── TPMTracker.wait_if_needed()   [if configured]
        ├── HistoryTrimmer.trim()         [if configured]
        ├── RetryHandler.execute(_call)  [if configured]
        ├── _call(messages, stream)       [transport]
        │     ├── OllamaBackend  → client.chat()
        │     └── OpenAIBackend  → openai.Client.chat.completions.create()
        └── TPMTracker.record_usage()    [if configured]
```

---

## Testing

The project includes 157 tests across three test files:

```bash
# Run all tests
python -m pytest test_parser.py test_session.py test_backends.py -v

# Run only backend tests (OllamaBackend, OpenAIBackend, middleware)
python -m pytest test_backends.py -v

# Run only session-level tests (feedback loop, streaming, interruption)
python -m pytest test_session.py -v

# Run only parser/executor tests
python -m pytest test_parser.py -v
```

### Test Coverage

| File | Tests | Scope |
|------|-------|-------|
| `test_parser.py` | 103 | Parser (WRITE/EDIT/FILE/RUN/TOOL/SKILL), matching, edit utils, action executor safety |
| `test_session.py` | 21 | Feedback loop, streaming, interruption handling, system prompt |
| `test_backends.py` | 33 | OllamaBackend, OpenAIBackend, TokenCounter, TPMTracker, RetryHandler, HistoryTrimmer, Backend.call() orchestration |

The session and backend tests use mock clients (`MockClient`,
`InterruptingClient`, `MockOllamaClient`, `MockOpenAIClient`) that
simulate API responses — no real Ollama or OpenAI server required.

---

## License

MIT
