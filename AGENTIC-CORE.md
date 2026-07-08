# Agentic Core — Architecture Assessment

## Purpose

A minimalistic, reusable Python package extracted from the
`ollama-chat-agentic` project. It implements a block-based action protocol
(WRITE / EDIT / FILE / RUN / TOOL / SKILL with EOF markers) that any Ollama
LLM chat application can embed to gain agentic file, shell, and tool
capabilities.

## Source Analysis

The original `ollama-chat-agentic` is ~3000+ lines across 20 modules with
persistence, jobs, MCP, memory, slash commands, spinners, and 30 tools.
The minimal core strips ~80% of that app-specific weight and keeps only
the **protocol engine + execution + feedback loop**.

Key source files studied:

| File | Role | Kept? |
|---|---|---|
| `parser.py` (436 lines) | Block extraction engine — regex-based, handles fence depth, bare EOF, markdown blocks | Yes, nearly intact |
| `actions.py` (1182 lines) | Block executor: WRITE/EDIT/FILE/RUN/TOOL/SKILL dispatch | Yes, condensed |
| `session.py` (959 lines) | Feedback loop: model → parse → execute → observe → repeat | Yes, simplified |
| `constants.py` | Limits, safety lists, ANSI colors, platform info | Yes, trimmed |
| `edit_utils.py` | Diff/summary helpers for EDIT | Yes, intact |
| `tools/__init__.py` | Tool base class + registry + built-in tools | Yes, trimmed to essentials |
| `skills/base.py` | Skill + PromptOnlySkill + MarkdownSkill | Partial — Skill + PromptOnlySkill only |

## Component Breakdown

```
agentic_core/
├── __init__.py          # Public API: AgenticSession, parse_actions, build_system_prompt
├── constants.py         # Limits, safety lists, ANSI colors, platform info
├── parser.py            # Block extraction engine (protocol parser)
├── system_prompt.py     # Modular system prompt builder
├── actions.py           # Block executor: WRITE/EDIT/FILE/RUN/TOOL/SKILL dispatch
├── session.py           # Feedback loop: model → parse → execute → observe → repeat
├── edit_utils.py        # Diff/summary helpers for EDIT
├── tools/
│   ├── __init__.py      # Tool base class + registry + built-in tools
│   └── fs.py            # Essential file tools (read/write/list/search)
└── skills/
    └── base.py          # Skill + PromptOnlySkill base classes
```

## Data Flow

```
User input
    │
    ▼
┌──────────┐    stream     ┌──────────┐
│  Ollama  │ ◄──────────►  │ Session  │  (feedback loop)
│  Client  │               │  (loop)  │
└──────────┘               └────┬─────┘
                                │ parse_actions(text)
                                ▼
                         ┌──────────┐
                         │  Parser  │ → [(type, path, code, lang, closed), ...]
                         └────┬─────┘
                              │ for each action
                              ▼
                    ┌──────────────────┐
                    │  ActionExecutor  │
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

## Key Decisions

| Decision | Rationale |
|---|---|
| **Keep parser.py nearly intact** | It's the protocol engine — 436 lines, self-contained, battle-tested regex logic. Rewriting risks introducing edge-case bugs (fence depth, bare EOF, markdown blocks). |
| **Single `actions.py`** | The original spreads logic across actions.py + matching.py + edit_utils.py + process.py. One executor module with clear methods is simpler to follow and embed. |
| **Tool registry as subpackage** | Tools are the primary extension point. Apps will add their own. Ship only 4-5 essential built-ins (read_file, write_file, list_files, search_in_files, run_command). |
| **Skills: thin base class only** | Keep Skill + PromptOnlySkill. Drop MarkdownSkill (script resolution, path rewriting, reference loading) — that's app-level. |
| **No persistence/jobs/MCP/memory/slash commands** | These are app-level concerns, not core protocol concerns. The core is stateless between turns except conversation history. |
| **Inject Ollama client via duck-typing** | Don't couple to `ollama` library. Accept any client with a `.chat()` method. Makes the core testable and embeddable. |
| **System prompt as builder function** | A builder function (`build_system_prompt(tools=True, skills=False, ...)`) lets apps customize sections, inject identity, and control what's included. |
| **Pluggable confirmation** | Default to auto-approving safe operations. Provide a `confirm_callback` hook for apps wanting interactive confirmation. |

## Technology Choices

- **Pure Python, zero hard dependencies.** The only external dependency is the
  Ollama client, which is *injected* — the core itself imports nothing beyond
  stdlib.
- **`subprocess`** for RUN — stdlib only, cross-platform.
- **`difflib`** for EDIT diffs — stdlib only.
- **`json`** for TOOL/SKILL params — stdlib only.
- No async — the feedback loop is inherently sequential.

## Public API

```python
from agentic_core import AgenticSession, build_system_prompt
from agentic_core.tools import Tool, register
from agentic_core.skills import Skill

session = AgenticSession(
    client=my_ollama_client,     # any object with .chat(model=, messages=, stream=)
    model="glm-5.1:cloud",
    workdir=".",
    tools=True,
    skills=True,
    system_prompt=build_system_prompt(name="myapp", tools=True, skills=True),
)
response_text = session.run("Create hello.py and run it")
```

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Parser edge cases (nested fences, bare EOF, smart quotes) | Keep original parser logic intact. Add test suite. |
| Infinite feedback loops | Hard limit `MAX_FEEDBACK_ROUNDS=3` per turn + identical-action detection. |
| Destructive shell commands | Blocked/warning command lists + pluggable `confirm_callback`. |
| Context bloat from large file reads | Per-observation and total-observation char limits with truncation. |
| Coupling to Ollama API shape | Accept client via duck-typing; document expected `.chat()` signature. |

## Extension Points

- **Custom tools**: Subclass `Tool`, implement `execute(params, workdir)`,
  call `register(MyTool())`. The tool's `system_prompt` auto-injects.
- **Custom skills**: Subclass `Skill`, set `system_prompt` + `parameters`,
  implement `execute()`.
- **Confirmation**: Pass `confirm_callback=lambda action_type, details: True`
  for auto-approve-all, or implement interactive logic.
- **System prompt**: Pass a custom `system_prompt` string to `AgenticSession`
  to override the default entirely.
