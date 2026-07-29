"""LLM Backend Abstraction Layer.

Provides unified interface for different LLM APIs:
- OllamaBackend: Wraps a duck-typed client with .chat(), zero middleware overhead
- OpenAIBackend: Uses openai.Client, auto-configures middleware based on tpm_limit

Optional middleware (OpenAI-compatible, None for Ollama):
- TokenCounter: tiktoken-based token counting with char/4 fallback
- HistoryTrimmer: client-side context trimming with summarization
- RetryHandler: jittered exponential backoff for transient errors
- TPMTracker: proactive tokens-per-minute throttling

When tpm_limit is set on OpenAIBackend, all four middleware components are
auto-configured as a coordinated bundle: aggressive trimming (smaller
context = fewer tokens per request), proactive TPM waiting, and 429-aware
retry. Without tpm_limit, trimming uses full ctx_size and retry is
connection-errors-only.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any

from .constants import MODEL_TEMPERATURE
from .utils import agent_print

logger = logging.getLogger(__name__)


# ── Token Counter ─────────────────────────────────────────────────────

class TokenCounter:
    """Counts tokens using tiktoken, with graceful fallback to char/4 heuristic."""

    def __init__(self, model: str, per_message_overhead: int = 4):
        self._encoder = None
        self.per_message_overhead = per_message_overhead
        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass  # Fallback to heuristic

    def count(self, messages: List[Dict]) -> int:
        """Count tokens in message list. Falls back to char/4 heuristic."""
        if self._encoder:
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(self._encoder.encode(content))
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            total += len(self._encoder.encode(part["text"]))
                total += self.per_message_overhead
            return total
        return sum(len(str(m.get("content", ""))) for m in messages) // 4

    def count_message(self, msg: Dict) -> int:
        """Count tokens in a single message."""
        return self.count([msg])


# ── TPM Tracker ───────────────────────────────────────────────────────

class TPMTracker:
    """Tracks tokens-per-minute usage to proactively throttle requests.

    Maintains a rolling 60-second window of token usage. Before sending,
    checks if the request would exceed the TPM budget. If so, waits until
    the window clears — transforming 429 errors into planned waits.
    """

    def __init__(self, tpm_limit: int, quiet: bool = False):
        self.tpm_limit = tpm_limit
        self.quiet = quiet
        self._usage_log: List[Tuple[float, int]] = []

    def _prune(self, now: float):
        """Remove entries older than 60 seconds."""
        cutoff = now - 60.0
        self._usage_log = [(t, c) for t, c in self._usage_log if t > cutoff]

    def _current_usage(self) -> int:
        """Return total tokens used in the last 60 seconds."""
        now = time.time()
        self._prune(now)
        return sum(c for _, c in self._usage_log)

    def wait_if_needed(self, estimated_tokens: int):
        """Block until sending estimated_tokens won't exceed TPM limit.

        If a single request exceeds the TPM limit, waiting is pointless —
        just send it and let the retry handler deal with any 429.
        """
        if estimated_tokens >= self.tpm_limit:
            return
        while True:
            current = self._current_usage()
            if current + estimated_tokens <= self.tpm_limit:
                return
            now = time.time()
            if self._usage_log:
                oldest_time = self._usage_log[0][0]
                wait_seconds = max(1, oldest_time + 60.0 - now)
            else:
                wait_seconds = 1
            if not self.quiet:
                agent_print(
                    f"\n[TPM limit: {current}/{self.tpm_limit} tokens used. "
                    f"Waiting {wait_seconds:.0f}s for budget to clear...]\n"
                )
            time.sleep(min(wait_seconds, 60))

    def record_usage(self, tokens: int):
        """Record actual token usage after a successful request."""
        self._usage_log.append((time.time(), tokens))


# ── History Trimmer ───────────────────────────────────────────────────

class HistoryTrimmer:
    """Trims conversation history to fit within a token budget.

    Uses summarization for dropped messages to preserve context anchors.
    Preserves tool-call/result pairs to avoid API rejection.
    """

    def __init__(self, token_counter: TokenCounter, max_tokens: int,
                 reserve_output: int = 2048):
        self.counter = token_counter
        self.max_tokens = max_tokens
        self.reserve_output = reserve_output

    @property
    def input_budget(self) -> int:
        return self.max_tokens - self.reserve_output

    def trim(self, messages: List[Dict]) -> List[Dict]:
        """Trim messages to fit within input_budget."""
        if not messages:
            return messages
        total = self.counter.count(messages)
        if total <= self.input_budget:
            return messages
        if len(messages) <= 2:
            return messages

        system_msg = messages[0]
        last_msg = messages[-1]
        middle = messages[1:-1]
        groups = self._group_tool_pairs(middle)

        dropped_summaries: List[str] = []
        while groups and self.counter.count(
            [system_msg] + self._flatten(groups) + [last_msg]
        ) > self.input_budget:
            group = groups.pop(0)
            dropped_summaries.append(self._summarize_group(group))

        result = [system_msg]
        if dropped_summaries:
            result.append({
                "role": "system",
                "content": f"[Earlier conversation trimmed — {len(dropped_summaries)} message(s) dropped: "
                           + "; ".join(dropped_summaries) + "]"
            })
        result += self._flatten(groups)
        result.append(last_msg)

        final_count = self.counter.count(result)
        if final_count > self.input_budget:
            overflow = final_count - self.input_budget
            if isinstance(last_msg.get("content"), str):
                content = last_msg["content"]
                keep_chars = max(100, len(content) - overflow * 4)
                result[-1] = dict(last_msg)
                result[-1]["content"] = content[:keep_chars] + "\n[... truncated to fit context limit ...]"

        return result

    def _group_tool_pairs(self, messages: List[Dict]) -> List[List[Dict]]:
        """Group tool-call and tool-result messages for atomic trimming."""
        groups = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            if role == "assistant" and msg.get("tool_calls"):
                group = [msg]
                i += 1
                while i < len(messages) and messages[i].get("role") == "tool":
                    group.append(messages[i])
                    i += 1
                groups.append(group)
            else:
                groups.append([msg])
                i += 1
        return groups

    def _flatten(self, groups: List[List[Dict]]) -> List[Dict]:
        return [msg for group in groups for msg in group]

    def _summarize_group(self, group: List[Dict]) -> str:
        parts = []
        for msg in group:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))
            parts.append(f"{role}: {content[:80]}")
        return " | ".join(parts)


# ── Retry Handler ─────────────────────────────────────────────────────

class RetryHandler:
    """Retry on rate limits and connection errors with jittered backoff."""

    def __init__(self, triggers=None, initial_wait=20, max_wait=60,
                 max_retries=3, quiet=False):
        self.triggers = triggers or ["429", "rate_limit", "connection"]
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.max_retries = max_retries
        self.quiet = quiet

    def _is_retryable(self, error: Exception) -> bool:
        err_str = str(error).lower()
        return any(t in err_str for t in self.triggers)

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Try to extract Retry-After header value from error."""
        try:
            if hasattr(error, 'response') and hasattr(error.response, 'headers'):
                retry_after = error.response.headers.get('retry-after')
                if retry_after:
                    try:
                        return float(retry_after)
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        from datetime import datetime, timezone
                        dt = parsedate_to_datetime(retry_after)
                        if dt:
                            return max(0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            pass
        return None

    def _cancellable_sleep(self, seconds: float):
        """Sleep with periodic status updates and Ctrl+C responsiveness."""
        for elapsed in range(int(seconds)):
            time.sleep(1)
            if not self.quiet and elapsed > 0 and elapsed % 5 == 0:
                agent_print(f"  Waiting... {int(seconds) - elapsed}s remaining\n")

    def execute(self, func, *args, **kwargs):
        """Execute func with retry. Returns result or raises."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not self._is_retryable(e) or attempt == self.max_retries:
                    raise
                wait = min(self.initial_wait * (2 ** attempt), self.max_wait)
                wait += random.uniform(0, 2)
                retry_after = self._extract_retry_after(e)
                if retry_after is not None:
                    wait = max(wait, retry_after)
                if not self.quiet:
                    agent_print(
                        f"\n[Retryable error. Waiting {wait:.0f}s before retry "
                        f"({attempt + 1}/{self.max_retries})...]\n"
                    )
                self._cancellable_sleep(wait)
                last_error = e
        raise last_error


# ── Backend ABC ───────────────────────────────────────────────────────

class Backend(ABC):
    """Abstract base class for LLM API backends.

    All middleware components are Optional and default to None.
    This means Ollama with no middleware is zero-overhead passthrough.

    The call() method orchestrates middleware in order:
        1. TPMTracker.wait_if_needed()  — proactive throttle
        2. HistoryTrimmer.trim()        — context management
        3. RetryHandler.execute(_call)  — retry on transient errors
        4. TPMTracker.record_usage()    — record actual usage

    For Ollama (all components None), call() is a direct passthrough
    to _call() with zero overhead.
    """

    def __init__(self):
        self._token_counter: Optional[TokenCounter] = None
        self._trimmer: Optional[HistoryTrimmer] = None
        self._retry: Optional[RetryHandler] = None
        self._tpm_tracker: Optional[TPMTracker] = None

    def call(self, messages: List[Dict], stream: bool = True,
             on_chunk=None) -> Tuple[str, Optional[int]]:
        """Common call flow. Components activate only if configured.

        Args:
            messages: Conversation history (list of message dicts).
            stream: Whether to stream the response.
            on_chunk: Callback(text) called for each streaming chunk.

        Returns:
            Tuple of (response_text, token_count_or_None).
            token_count is used for TPM recording.

        Note: Does NOT catch KeyboardInterrupt — it propagates to the
        session which raises AgenticInterrupted with partial text.
        """
        # 1. TPM check (only if configured)
        if self._tpm_tracker and self._token_counter:
            estimated = self._token_counter.count(messages)
            self._tpm_tracker.wait_if_needed(estimated)

        # 2. Trim history (if configured)
        trimmed = self._trimmer.trim(messages) if self._trimmer else messages

        # Count tokens for TPM recording (post-trim)
        final = None
        if self._token_counter:
            final = self._token_counter.count(trimmed)
            if self._trimmer:
                original = self._token_counter.count(messages)
                if original != final:
                    logger.info("History trimmed: %d → %d tokens", original, final)

        # 3. Execute with retry or direct call
        if self._retry:
            text, tokens = self._retry.execute(
                self._call, trimmed, stream=stream, on_chunk=on_chunk
            )
        else:
            text, tokens = self._call(trimmed, stream=stream, on_chunk=on_chunk)

        # 4. Record TPM usage (use actual tokens if available, else estimate)
        if self._tpm_tracker:
            record = tokens if tokens is not None else final
            if record is not None:
                self._tpm_tracker.record_usage(record)

        return text, tokens

    @abstractmethod
    def _call(self, messages: List[Dict], stream: bool = True,
              on_chunk=None) -> Tuple[str, Optional[int]]:
        """Backend-specific API call. Returns (text, token_count_or_None).

        Must call on_chunk(text) for each streaming piece if stream=True.
        Must NOT catch KeyboardInterrupt — let it propagate.
        """
        pass


# ── Ollama Backend ────────────────────────────────────────────────────

class OllamaBackend(Backend):
    """Thin wrapper around a duck-typed Ollama-compatible client.

    Zero middleware overhead — all components are None. This preserves
    the exact behavior of agentic_core's existing client.chat() calls.

    The client is injected (not imported), maintaining the core's
    dependency-free design. Any object with .chat(model=, messages=,
    stream=, options=) works.
    """

    def __init__(self, client, model, temperature=0.0):
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature
        # All middleware = None → zero overhead, exact current behavior

    def _call(self, messages: List[Dict], stream: bool = True,
              on_chunk=None) -> Tuple[str, Optional[int]]:
        """Call the model via the injected client.

        Handles both dict-style and object-style chunk formats.
        Does NOT catch KeyboardInterrupt — propagates to session.
        """
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options={"temperature": self.temperature},
        )

        if stream:
            chunks = []
            eval_count = None
            for chunk in response:
                text = ""
                if isinstance(chunk, dict):
                    text = chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        eval_count = chunk.get("prompt_eval_count")
                elif hasattr(chunk, "message"):
                    text = chunk.message.content
                if text:
                    chunks.append(text)
                    if on_chunk:
                        on_chunk(text)
            return "".join(chunks), eval_count
        else:
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
                return content, response.get("prompt_eval_count")
            content = response.message.content
            return content, getattr(response, "prompt_eval_count", None)


# ── OpenAI Backend ─────────────────────────────────────────────────────

class OpenAIBackend(Backend):
    """OpenAI-compatible API backend using openai.Client.

    Works with any OpenAI-compatible endpoint: OpenAI, Ollama /v1,
    vLLM, LM Studio, etc.

    Auto-configures middleware based on tpm_limit:
        Without tpm_limit: TokenCounter + HistoryTrimmer(full ctx) + RetryHandler(light)
        With tpm_limit:    TokenCounter + HistoryTrimmer(capped) + RetryHandler(aggressive) + TPMTracker

    The coupling is intentional: enabling TPM means "I'm on a rate-limited
    endpoint, so trim aggressively, wait proactively, and retry reactively."
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 ctx_size: int = 8192, temperature: float = 0.0,
                 thinking: bool = False, tpm_limit: Optional[int] = None,
                 max_context: Optional[int] = None, quiet: bool = False):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI backend. "
                "Run: pip install openai"
            )

        # Ensure /v1 suffix
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.ctx_size = ctx_size
        self.temperature = temperature
        self.thinking = thinking
        self._quiet = quiet

        # Token counter — always on for OpenAI-compatible backends
        self._token_counter = TokenCounter(model, per_message_overhead=4)

        # History trimmer — always on (OpenAI protocol doesn't guarantee
        # server-side eviction). Budget depends on TPM:
        #   Without --tpm: use full ctx_size
        #   With --tpm: cap at max_context (default 16384) — smaller context
        #   means fewer tokens per request, stretching the TPM budget
        trim_budget = ctx_size
        if tpm_limit is not None:
            trim_budget = min(ctx_size, max_context or 16384)
        self._trimmer = HistoryTrimmer(
            token_counter=self._token_counter,
            max_tokens=trim_budget,
            reserve_output=2048,
        )

        # Retry + TPM — coupled by design
        if tpm_limit is not None:
            # Aggressive: 429 + rate limits, 20s backoff
            self._retry = RetryHandler(
                triggers=["429", "rate_limit", "connection"],
                initial_wait=20,
                max_wait=60,
                max_retries=3,
                quiet=quiet,
            )
            self._tpm_tracker = TPMTracker(tpm_limit=tpm_limit, quiet=quiet)
        else:
            # Light: connection/500 only, 5s backoff, no 429 logic
            self._retry = RetryHandler(
                triggers=["connection", "500", "internal server error"],
                initial_wait=5,
                max_wait=30,
                max_retries=3,
                quiet=quiet,
            )
            self._tpm_tracker = None

    @staticmethod
    def _sanitize_messages(messages: List[Dict]) -> List[Dict]:
        """Convert messages to OpenAI format."""
        sanitized = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # Already in multimodal format (vision)
                sanitized.append(m)
            else:
                sanitized.append({
                    "role": m.get("role", "user"),
                    "content": content,
                })
        return sanitized

    def _build_create_kwargs(self, messages: List[Dict], stream: bool) -> Dict[str, Any]:
        """Build kwargs for chat.completions.create."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        # Some models (e.g. o1/o3) don't support max_completion_tokens.
        # We try it first, fall back to max_tokens, then to no max param.
        max_val = min(self.ctx_size, 8192)
        kwargs["max_completion_tokens"] = max_val
        return kwargs

    def _call_create(self, messages: List[Dict], stream: bool):
        """Call create with fallback for max_tokens parameter incompatibility."""
        kwargs = self._build_create_kwargs(messages, stream)
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "max_completion_tokens" in err_str or "unrecognized" in err_str:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = min(self.ctx_size, 8192)
                try:
                    return self.client.chat.completions.create(**kwargs)
                except Exception as e2:
                    err_str2 = str(e2).lower()
                    if "max_tokens" in err_str2 or "unsupported" in err_str2:
                        kwargs.pop("max_tokens", None)
                        return self.client.chat.completions.create(**kwargs)
                    raise
            if "unsupported" in err_str and "max" in err_str:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = min(self.ctx_size, 8192)
                try:
                    return self.client.chat.completions.create(**kwargs)
                except Exception as e2:
                    if "max_tokens" in str(e2).lower() or "unsupported" in str(e2).lower():
                        kwargs.pop("max_tokens", None)
                        return self.client.chat.completions.create(**kwargs)
                    raise
            raise

    def _call(self, messages: List[Dict], stream: bool = True,
              on_chunk=None) -> Tuple[str, Optional[int]]:
        """Call the model via OpenAI SDK.

        Handles max_completion_tokens → max_tokens → no-max fallback.
        Does NOT catch KeyboardInterrupt — propagates to session.
        """
        messages = self._sanitize_messages(messages)

        if stream:
            return self._call_streaming(messages, on_chunk)
        return self._call_blocking(messages)

    def _call_streaming(self, messages: List[Dict],
                        on_chunk=None) -> Tuple[str, Optional[int]]:
        """Stream response using OpenAI SDK."""
        response = self._call_create(messages, stream=True)
        chunks = []
        eval_count = None

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content or ""
            if text:
                chunks.append(text)
                if on_chunk:
                    on_chunk(text)

        # Get usage from final chunk if available
        if hasattr(response, 'usage') and response.usage:
            eval_count = response.usage.prompt_tokens

        return "".join(chunks), eval_count

    def _call_blocking(self, messages: List[Dict]) -> Tuple[str, Optional[int]]:
        """Non-streaming call using OpenAI SDK."""
        response = self._call_create(messages, stream=False)
        content = response.choices[0].message.content or ""
        eval_count = response.usage.prompt_tokens if response.usage else None
        return content, eval_count
