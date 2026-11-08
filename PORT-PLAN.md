# Port Plan: uhu Native Tool_calls → agentic-core

Source: `/Users/andreis/Projects/GitHub/uhu` commits `6dd6e0e..3dc787c` (2026-11-08).
Target: `agentic_core` in this repo. Constraint: zero hard dependencies, stdlib only.

Reference docs (do not copy verbatim — adapt):
- uhu `dev-docs/NATIVE-TOOL-CALLS-ARCH.md` (field fixes 1–7.5)
- uhu `dev-docs/NATIVE-TOOLS-SWITCH-ARCH.md` (tools-mode design)
- uhu `dev-docs/AS-IS-HISTORY-ARCH.md` (deferred — needs own design pass)

---

## Phase 1 (P1) — Native tool_calls capture in backends

**Scope:** `agentic_core/backends.py`, `agentic_core/tool_calls.py`
**Source:** uhu `6dd6e0e` backends.py deltas, `tool_calls.py`

1.1 **`tool_calls.py` completion.** Port the parts the core copy lacks (diff = ~96 lines):
   - `_MAX_NATIVE_NAME_LEN = 64` blob guard → degrade to `_malformed_native_call` with `_raw` 512-char preview
   - `_TEXT_ACTION_MARKER_RE` + `looks_like_text_action` → steer `_text_action_mixing` entries
   - `_MARKER_NAME_RE` extended beyond TOOL/SKILL wrappers (keep `_clean_native_name` as single choke point)
   - `_STRUCTURED_PARAM_KEYS` coercion (already present in core — verify parity)

1.2 **Backend contract.** Add to `Backend` ABC docs + both implementations:
   - `self.last_tool_calls: [{"name": str, "params": dict}]`, reset per call, set in
     streaming AND blocking paths on both backends
   - `call()` contract unchanged: empty content stays empty (no synthesis in backend — transparency rule)
   - `OllamaBackend`: chunk-level `message.tool_calls` collection + defensive fragment merge
   - `OpenAIBackend`: `choices[0].delta.tool_calls` delta fragments (`index`-keyed concat),
     `finish_reason=="tool_calls"` detection, `arguments` JSON-string parsing,
     `reasoning`→thinking mapping
   - uhu's Pydantic `extra='allow'` Message patch is Ollama-host-specific — skip unless core needs it

1.3 **CLI/config nudge.** Core has no CLI; expose constructor-level only
   (`AgenticSession(native_tools=None/True/False)`, default None = "when backend supports it").
   uhu's `--tools-mode` tri-state maps to: session param `tools_mode: {"native","hybrid","text"}`.

**Tests:** port streaming-fragment tests + malformed-name cases from uhu
`tests/test_tool_calls.py` (subset ~300 lines) into `tests/test_tool_calls.py`.
Mock backends emitting structured tool_calls, no real server.

**Exit:** `pytest tests/test_tool_calls.py` green; text-channel regression suite (202 tests) unchanged-green.

---

## Phase 2 (P1) — Pipeline merge + protocol correctness

**Scope:** `agentic_core/actions.py`, `agentic_core/session.py`
**Source:** uhu `d9abbd8`, `10df9a9`, `bc74abc` session/actions deltas

2.1 **Merge in executor.** After text-parsed actions, append native calls as
   `{"type":"tool", "closed":True, "native":True}`. Order: text actions first, then native.
   - Dedupe by `(name, params-signature)` — models occasionally emit both channels
   - `params_explicit: True` on native entries (u Field fix #2: parameterless calls
     must not hit the empty-params truncation warning)
   - Identical-action loop guard (`MAX_IDENTICAL_ACTION_REPEATS`) counts native signatures

2.2 **Approval parity.** Native tool calls pass the same `confirm_callback` gate as
   `**TOOL:**` blocks. `auto_approve` per-tool semantics unchanged.

2.3 **`_wire_tool_results`** (backends, wire boundary only — stored history untouched):
   convert the user-observation message following a `tool_calls` assistant into
   per-call `tool`-role results; defer intervening system messages; on /v1
   synthesize `tool_call_id`s. Fixes 400s on native-function-calling providers
   (mistral-class). Both backends.

**Tests:** approval-parity tests, merge/dedupe tests, wire-conversion test
(uhu `test_tool_calls.py` "wire" section, ~150 lines).

**Exit:** mixed text+native and native-only rounds 1–3 verified on scripted backends
(port uhu `test_native_feedback_loop.py` shapes — 5 scripted backend classes, ~200 lines).

---

## Phase 3 (P2) — Loop robustness (session)

**Scope:** `agentic_core/session.py`
**Source:** uhu `bc74abc` (+ its 155-line test), `10df9a9` wrap-up fix

3.1 **Cross-send repetition radar.** Port as-is, minus uhu host specifics:
   - `_cross_sig_rounds` persistent across `.run()` invocations, reset on NEW user message
   - warn at `CROSS_SEND_LOOP_WARN_THRESHOLD=3` (constant → `constants.py`)
   - decay by 2 on disk-change fingerprint between identical prompts
     (uhu uses workdir fingerprint — port `_workdir_fingerprint` lite version)
   - soft warning only; hard skip stays within-run

3.2 **Wrap-up/loop gating.** Not directly portable — uhu's wrap-up is session-RPC host
   logic. Port the portable kernel: `_is_warning_only_observation` classifier +
   the rule "grace round fires only for warning-shaped observations".
   Loop nudges: core feeds observations as user messages today; add console-visibility
   counterpart only if host passes `on_chunk`/echo callback.

**Tests:** port uhu `test_cross_send_radar.py` (~155 lines, adapt to AgenticSession.run loop).

**Exit:** scripted "same prompt ×3 → same native read" produces one soft warning, not skip;
different prompt resets.

---

## Phase 4 (P2) — Display/render fidelity (optional, host-visible)

**Scope:** `agentic_core/session.py` (on_chunk path)
**Port:** uhu findings #4–6 console side, minimal variant:
   - native render `[native tool_call N/M: name]` + args preview (caps: 600 console,
     160 per value line; full to log only) — core has no console, so render goes
     through `on_chunk`
   - render recovery `_recover_native_from_render` + flag-gated loss detector
     (`_native_rendered_last_call` set only when core itself rendered)
   - render-echo breaker (SHAPE-gated) as a SYSTEM WARNING observation
   - de-formatted warning texts (never quote the literal format)

**Decision point:** if agentic-core keeps display out of scope (host renders), ship
only the recovery + breaker + detector (they operate on response text, not console).

**Tests:** port from uhu `test_wrapup_grace_gating.py` the render/breaker cases only.

---

## Phase 5 (P3) — As-is history (deferred, needs design pass)

uhu's AS-IS-HISTORY-ARCH stores assistant turns as raw structured
`{content, tool_calls:` entries instead of render text. **Do not port now:**
   - changes `AgenticSession.history` shape (consumers expect `{"role","content"}` — README-documented API)
   - interacts with `HistoryTrimmer._group_tool_pairs` and session persistence
   - mimicry risk in core is lower (core does not store renders unless host echoes)

Revisit when Phase 4 ships, with a version marker + backward-compat rule
(old sessions load unchanged).

---

## Phase 6 (P3) — Smaller items, opportunistic

- Flood guard neutralizer (`neutralize_screen_controls`): port to `utils.py` +
  apply to RUN observation echo only when core echoes (host decision). ~40 lines.
- model_test tool: core does not ship `model_test` (host tool in uhu). Skip.
- `py_compile` SAFE_TOOLS removal: N/A — core has no py_compile in SAFE lists; verify.

## Not ported (explicit)

- uhu-only tools: browser, google_calendar, jobs, ui_automation, android_build,
  npm_build, pytest_tool, sys_info, weather, git, file_link, env_info,
  llm_query/llm_chat, MCP tools, run_command — host-level, against core's
  minimalism (u README: education/creativity, ~5K LOC budget).
- `--tools-mode` CLI flag machinery: core has no CLI; expose as constructor params.
- Capability auto-detection via `/api/tags`: host concern (uhu probes server state;
  core is transport-agnostic). Provide `Backend.supports_tools: bool|None` attribute
  instead; host sets it.

---

## Ordering & budget

| Phase | LOC (est.) | Tests (est.) | Risk |
|---|---|---|---|
| 1 | ~250 | ~300 | low (no API break) |
| 2 | ~200 | ~350 | medium (approval + wire paths) |
| 3 | ~120 | ~160 | low |
| 4 | ~150 | ~100 | medium (renders in core) |
| 5 | — | — | deferred |
| 6 | ~40 | — | trivial |

Phases 1–2 are the spine: land them as one PR. Phases 3–4 independent, can follow
separately. Keep core at ~6K LOC ceiling. Run full suite after each phase;
`MODELS-TEST.md`-style live validation on a native-calling model (beam:cloud /
mistral-large-4:cloud via OpenAIBackend) after Phase 2.
