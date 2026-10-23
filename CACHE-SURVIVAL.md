# KV Cache Survival in Ollama — Interleaved Conversations

How to keep Ollama's prompt-prefix KV cache alive when multiple applications
talk to the same model.

## The Problem

Ollama caches the KV (attention) state of the most recently processed prompt
**per loaded model** — by default a single slot. Caching is implicit: a new
request reuses cached state only for the token prefix it shares with the
previously processed sequence.

Scenario:

```
t1: app1 turn N      → slot holds app1's sequence
t2: app2 request     → no shared prefix → slot overwritten with app2's sequence
t3: app1 turn N+1    → compares against app2's sequence → ~0 cache hits
```

If app1 and app2 have **totally different system prompts**, the common prefix
is empty (BOS only). Every turn after an interleaved request re-evaluates the
entire prompt — system prompt + full history — at prompt-eval speed. For long
agentic conversations on local hardware this is the worst case: seconds to
minutes of prompt processing per turn.

Cache hits are visible in the response metadata:

```json
"prompt_eval_cached_count": 4096,   // tokens reused from cache
"prompt_eval_count": 12,            // tokens actually evaluated
```

`prompt_eval_cached_count ≈ 0` on every turn = the cache is being displaced.

---

## Mitigations

### 1. Distinct model tags per app (most reliable)

Give each app its own tag pointing at the same weights:

```bash
ollama cp model1 model1-app1
ollama cp model1 model1-app2
```

Ollama treats each tag as a separate model with its own cache slot. App1 and
app2 no longer displace each other. Weights on disk are shared (blobs are
content-addressed), so this costs no extra disk space.

**Tradeoff:** each tag is a separate *loaded instance*. If both apps run
concurrently, both instances are resident → ~2× VRAM/RAM. If usage is
alternating with a reasonable `keep_alive`, only one is loaded at a time.

**Best for:** two or three long-lived apps with disjoint conversations.

### 2. `OLLAMA_NUM_PARALLEL` (multiple slots)

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

Allocates N sequence slots per model, each with its own KV cache. App1's
sequence can survive in its slot while app2 uses another.

**Reality check (verified in practice):** two parallel conversations on the
same model retain ~99% cache hits. Slot selection is **prefix-based**: the
runner picks the slot with the longest common prefix with the incoming
prompt, so each conversation keeps re-finding "its" slot. Assignment is
effectively sticky per conversation — displacement happens only when
conversations **exceed the slot count** (N+1 conversations on N slots),
not on ordinary interleaving.

**Caveats:**

- **Capacity, not interleaving, is the limit.** With more concurrent
  conversations than slots, the odd one out loses its slot and pays full
  prompt-eval on its next turn.
- **Sticky-by-prefix is emergent, not contractual.** It follows from
  longest-common-prefix slot selection; a doc-level guarantee could change
  across Ollama versions.
- **Context is divided.** `num_ctx` is split across slots: with
  `OLLAMA_NUM_PARALLEL=2` and `num_ctx=8192`, each request effectively gets
  ~4096 tokens of context. Long agentic histories may overflow and trim.
- KV memory is pre-allocated for all slots, increasing baseline VRAM usage.
- Default is auto-selected (1 or 4) based on available memory — set it
  explicitly if you depend on it.

**Best for:** moderate concurrency where some cache loss is acceptable and
VRAM is plentiful.

### 3. Serialize access (zero cost, zero risk)

Don't interleave: route requests through a shared queue or file lock so only
one app talks to the model at a time, and each app finishes its burst before
the next starts.

```python
# crude cross-process lock
lock = open("model1.lock", "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX)   # POSIX
    response = session.run(user_text)
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
```

(On Windows use `msvcrt.locking` or an OS-level mutex.)

**Tradeoff:** adds latency for the waiting app; unsuitable for interactive
apps that must respond immediately.

**Best for:** batch workloads, CI jobs, scheduled agents.

### 4. Hosted / OpenAI-compatible endpoints

Hosted providers implement prefix caching with per-account retention that
survives interleaving:

- **OpenAI:** automatic prefix caching, no configuration; cached prefix
  typically survives minutes of idle time and other traffic.
- **Anthropic:** explicit `cache_control` breakpoints with a 5-minute TTL
  (refreshable on each hit).
- **vLLM / LM Studio / other local servers:** vLLM does prefix caching per
  engine with better retention than Ollama's single-slot scheme.

agentic-core's `OpenAIBackend` benefits automatically — no code changes.

**Tradeoff:** cost per token, network latency, data leaves the machine.

### 5. Supporting settings (necessary, not sufficient)

These don't fix displacement but prevent *needless* cache loss:

- **`keep_alive`** — controls how long model *weights* stay loaded, not the
  KV cache. But if the model unloads between turns, the KV cache is lost
  with it. Set `keep_alive` ≥ your inter-turn gap (e.g. `"30m"` or `-1`)
  so the cache at least *can* survive.
- **`OLLAMA_KV_CACHE_TYPE=q8_0`** (requires flash attention) — halves KV
  memory, freeing room for longer contexts or more parallel slots at
  negligible quality cost.
- **Stable system prompts** — any byte-level change (date, dynamic content)
  invalidates the prefix from that point. agentic-core injects the current
  date into the system prompt; it changes only daily, so within a day the
  prefix is stable.

### 6. Watch out: history trimming breaks prefix reuse

`HistoryTrimmer` (used by `OpenAIBackend`, especially with `tpm_limit`)
drops oldest messages. Any trim rewrites the middle of the sequence and
invalidates the cached prefix from that point onward — on Ollama-style
prefix caching this converts a full-cache-hit turn into a partial miss.

If you run `OpenAIBackend` against a local Ollama `/v1` endpoint, prefer
generous `ctx_size` and avoid aggressive `max_context` caps unless TPM
budgeting actually requires it.

---

## API Surface vs Inference Engine

Cache behavior is a property of the **engine**, not the API protocol.
The same model behind different API surfaces behaves differently:

| Endpoint | Engine | Caching semantics |
|---|---|---|
| `http://localhost:11434/api/chat` (native) | Ollama | Single-slot prefix cache — displacement applies |
| `http://localhost:11434/v1/chat/completions` (OpenAI-compatible) | Ollama | **Identical to native** — `/v1` is a thin translation layer |
| `api.openai.com/v1` | OpenAI fleet | Fleet prefix caching — multiple prefixes coexist on different machines |
| vLLM `/v1` | vLLM | Radix-cache prefix tree — multiple prefixes coexist |
| `*:cloud` models via local Ollama | ollama.com servers | Fleet prefix caching — local proxy only; interleaving-safe |

### OpenAI hosted (api.openai.com)

- **Automatic** for supported models on prompts > 1024 tokens; no session ID
- Requests are routed to servers that recently processed the same prefix —
  app1 and app2 prefixes are cached on **different machines simultaneously**,
  so interleaving does **not** cause displacement
- Cached input tokens cost up to ~90% less; latency reduction up to ~80%
- TTL ~5–10 minutes of inactivity (longer off-peak), refreshed on each hit
- Visibility: `usage.prompt_tokens_details.cached_tokens`
- Optional `prompt_cache_key` parameter improves routing affinity
- Prefix must be identical from byte 0; cache boundaries in 1024-token steps

### Ollama cloud models (`*:cloud`)

Cloud models (e.g. `glm-5.3-flash:cloud`) never load locally — the local
Ollama server is only a proxy. Inference and caching happen on ollama.com's
infrastructure, which uses fleet-style prefix caching like OpenAI: parallel
conversations on the same cloud model coexist without displacing each other
(verified in practice: ~99% cache hits with two interleaved conversations).

Consequences:

- Local `OLLAMA_NUM_PARALLEL` / `keep_alive` are irrelevant for cloud models
- `/api/ps` shows nothing while a cloud conversation is active
- Cache metrics are not exposed through the local proxy — judge by response
  latency (consistently fast turns = cache hits)

### Ollama's `/v1` endpoint

The OpenAI-compatible endpoint changes **syntax only**, not caching:

- Same single-slot displacement as the native API
- `prompt_cache_key` is ignored
- No `cached_tokens` in usage — check the native API's
  `prompt_eval_cached_count` (or server logs) instead
- No per-token discount — the "cost" is prompt-eval time, paid in full

**Rule of thumb:** `OpenAIBackend` pointed at `http://localhost:11434`
gets you OpenAI *syntax*, not OpenAI *caching*. All mitigations in this
document still apply. Pointed at api.openai.com, caching works
automatically — keep the system prompt byte-stable, put stable content
first, and prefer a generous `ctx_size` over aggressive trimming
(see warning above).

### Practical guidance for multi-agent workloads

**Cloud models (`*:cloud`):** run as many agents as needed — parallel
sessions, sub-agents, multi-agent orchestration. Caching is ollama.com's
concern and is interleaving-safe; the local `OLLAMA_NUM_PARALLEL` does not
apply (the local server only proxies).

**Local models:** agent count should not exceed slot count. With
`OLLAMA_NUM_PARALLEL:1` (the default on many installs), two concurrent
agents on the same local model displace each other's cache — every turn
pays full prompt-eval. Options:

- Raise `OLLAMA_NUM_PARALLEL` to 2–4 (costs VRAM, divides `num_ctx` per slot)
- Distinct model tags per agent (`ollama cp model model-agent1`)
- Accept the re-eval cost when histories are short

**Caveat:** cloud caching is server-side and undocumented — observed
behavior (~99% hits) is empirical, not contractual. If retention policy
changes, the effect is graceful (slower turns), not breakage.

---

## Decision Guide

| Situation | Mitigation |
|---|---|
| 2–3 apps, long conversations, enough VRAM | Distinct model tags (#1) |
| Many apps, short bursts, VRAM tight | `OLLAMA_NUM_PARALLEL` (#2) |
| Batch/CI workloads, latency tolerant | Serialize (#3) |
| Cloud/remote acceptable | Hosted endpoint (#4) |
| Single app, gaps between turns | `keep_alive` (#5) only |

## Verifying

Check cache behavior per request:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "model1", "keep_alive": "30m",
  "messages": [{"role": "user", "content": "ping"}]
}' | python -c "import json,sys; r=json.load(sys.stdin); \
print('cached:', r['prompt_eval_cached_count'], 'evaluated:', r['prompt_eval_count'])"
```

Healthy agentic loop: `cached` grows with history, `evaluated` stays small
(≈ the new turn's tokens). `cached ≈ 0` on every turn = displacement.
