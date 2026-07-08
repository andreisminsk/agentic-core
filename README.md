# agentic_core

A minimalistic, reusable Python package that gives any Ollama LLM chat
application agentic capabilities — file reading/writing/editing, shell
command execution, and extensible tool/skill workflows — through a
text-based block protocol.

## What This Is

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
The Ollama client is injected via duck-typing — the core never imports
the `ollama` library directly.

### Why Use It

- **Embed agentic behavior** in any Ollama chat app with ~3 lines of code
- **Zero dependencies** — pure Python stdlib, cross-platform
- **Extensible** — add custom tools and skills by subclassing
- **Safe** — built-in command safety checks, blocked/warning lists,
  pluggable confirmation callbacks
- **Context-aware** — automatic truncation prevents context window bloat
- **Battle-tested parser** — handles fence depth, bare EOF, smart quotes,
  markdown blocks, and edge cases from real LLM output

## Quick Start

### Installation

No install needed — just copy the `agentic_core/` package into your
project, or add it to your `PYTHONPATH`.

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
    workdir=".
    confirm_callback=confirm,
)
```

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
│ (chat panel) │◄────│ (bridge)     │◄────│ (workdir=     │
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

## API Reference

### AgenticSession

```python
AgenticSession(
    client,                  # Ollama-compatible client (duck-typed)
    model,                   # Model name string (e.g. "glm-5.1:cloud")
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

**Methods:**
- `.run(user_text)` → str: Run the feedback loop, return final prose
- `.reset()`: Clear history (keep system prompt)

**Attributes:**
- `.history`: List of `{"role": ..., "content": ...}` messages
- `.executor`: The `ActionExecutor` instance
- `.system_prompt`: The active system prompt string

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

### build_system_prompt

```python
build_system_prompt(
    name="uhu",
    tools=False,
    skills=False,
    extra_sections=None,  # list of strings to append
    identity=None,        # replace the entire base prompt
)
```

## Built-in Tools

| Tool | Auto-approve | Description |
|------|:---:|-------------|
| `read_file` | ✅ | Read file contents (offset/limit) |
| `write_file` | ✅ | Create/overwrite/append file |
| `replace_in_file` | ✅ | Exact string replacement |
| `list_files` | ✅ | List directory (recursive, glob) |
| `find_file` | ✅ | Find files by name pattern |
| `search_in_files` | ✅ | Regex search across files |
| `peek_file` | ✅ | Head + tail of a file |
| `mkdir` | ✅ | Create directory tree |
| `copy_file` | ✅ | Copy file or directory |
| `move_file` | ✅ | Move/rename file or directory |
| `web_search` | ❌ | Search the web via DuckDuckGo (needs `ddgs`) |
| `web_fetch` | ✅ | Fetch URL content, optional LLM summary |
| `http_request` | ❌ | HTTP GET/POST/PUT/DELETE (needs `httpx`) |
| `time_now` | ✅ | Current date/time/timezone |
| `calculator` | ✅ | Safe math expression evaluation |

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

## Project Structure

```
agentic_core/
├── __init__.py          # Public API
├── constants.py         # Limits, safety lists, platform info
├── parser.py            # Block extraction engine
├── system_prompt.py     # System prompt builder (with date/time injection)
├── actions.py           # Block executor + confirmation system
├── matching.py          # Fuzzy matching for EDIT
├── edit_utils.py        # Diff/summary helpers
├── session.py           # Feedback loop
├── tools/
│   ├── __init__.py      # Tool registry + base class
│   ├── fs.py            # Built-in file tools (10)
│   ├── web.py           # web_search, web_fetch, http_request, time_now
│   └── calculator.py     # Safe math expression evaluator
└── skills/
    ├── __init__.py      # Skill registry
    └── base.py          # Skill + PromptOnlySkill
```

## License

MIT
