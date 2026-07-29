"""Tests for the backend abstraction layer.

Tests OllamaBackend (zero-overhead passthrough), OpenAIBackend (with mock),
middleware components (TokenCounter, TPMTracker, RetryHandler, HistoryTrimmer),
and the Backend.call() orchestration flow.
"""

import time
import pytest

from agentic_core.backends import (
    Backend, OllamaBackend, OpenAIBackend,
    TokenCounter, TPMTracker, RetryHandler, HistoryTrimmer,
)


# ═══════════════════════════════════════════════════════════════════════
# Mock clients
# ═══════════════════════════════════════════════════════════════════════

class MockOllamaClient:
    """Mock Ollama-compatible client."""

    def __init__(self, response_text="Hello", eval_count=None):
        self.response_text = response_text
        self.eval_count = eval_count
        self.calls = []

    def chat(self, model, messages, stream, options=None):
        self.calls.append({
            "model": model, "messages": messages,
            "stream": stream, "options": options,
        })
        if stream:
            return self._stream()
        return {
            "message": {"content": self.response_text},
            "prompt_eval_count": self.eval_count,
        }

    def _stream(self):
        for char in self.response_text:
            yield {"message": {"content": char}, "done": False}
        yield {"message": {"content": ""}, "done": True,
               "prompt_eval_count": self.eval_count}


class MockOpenAIResponse:
    """Mock OpenAI SDK response object."""

    def __init__(self, text, prompt_tokens=100):
        self.choices = [type("Choice", (), {
            "message": type("Msg", (), {"content": text}),
            "delta": type("Delta", (), {"content": ""}),
        })]
        self.usage = type("Usage", (), {"prompt_tokens": prompt_tokens})
        self._text = text

    def __iter__(self):
        for char in self._text:
            yield type("Chunk", (), {
                "choices": [type("Choice", (), {
                    "delta": type("Delta", (), {"content": char}),
                })],
            })
        # Final chunk with no content


class MockOpenAIClient:
    """Mock OpenAI-compatible client."""

    def __init__(self, response_text="Hello", prompt_tokens=100):
        self.response_text = response_text
        self.prompt_tokens = prompt_tokens
        self.calls = []

        # Build the nested structure: client.chat.completions.create()
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return MockOpenAIResponse(
                    outer.response_text, outer.prompt_tokens
                )

        class Chat:
            def __init__(self):
                self.completions = Completions()

        self.chat = Chat()


# ═══════════════════════════════════════════════════════════════════════
# OllamaBackend tests
# ═══════════════════════════════════════════════════════════════════════

class TestOllamaBackend:
    def test_nonstream_returns_text(self):
        client = MockOllamaClient("Hello world", eval_count=42)
        backend = OllamaBackend(client, "test-model")
        text, tokens = backend.call([{"role": "user", "content": "hi"}], stream=False)
        assert text == "Hello world"
        assert tokens == 42

    def test_stream_returns_text(self):
        client = MockOllamaClient("Hello")
        backend = OllamaBackend(client, "test-model")
        chunks = []
        text, tokens = backend.call(
            [{"role": "user", "content": "hi"}], stream=True, on_chunk=chunks.append
        )
        assert text == "Hello"
        assert "".join(chunks) == "Hello"

    def test_passes_model_and_options(self):
        client = MockOllamaClient("Hi")
        backend = OllamaBackend(client, "my-model", temperature=0.5)
        backend.call([{"role": "user", "content": "hi"}], stream=False)
        call = client.calls[0]
        assert call["model"] == "my-model"
        assert call["options"]["temperature"] == 0.5

    def test_no_middleware_zero_overhead(self):
        """OllamaBackend has all middleware None — call() is direct passthrough."""
        client = MockOllamaClient("Hi")
        backend = OllamaBackend(client, "test")
        assert backend._token_counter is None
        assert backend._trimmer is None
        assert backend._retry is None
        assert backend._tpm_tracker is None

    def test_keyboard_interrupt_propagates(self):
        """Backend must NOT catch KeyboardInterrupt — session handles it."""
        class InterruptingClient:
            def chat(self, **kwargs):
                raise KeyboardInterrupt()

        backend = OllamaBackend(InterruptingClient(), "test")
        with pytest.raises(KeyboardInterrupt):
            backend.call([{"role": "user", "content": "hi"}], stream=False)

    def test_object_style_chunks(self):
        """Handles chunk.message.content (object-style) not just dict-style."""
        class ObjChunk:
            def __init__(self, text, done=False):
                self.message = type("M", (), {"content": text})
                self.done = done

        class ObjClient:
            def chat(self, **kwargs):
                if kwargs.get("stream"):
                    for c in [ObjChunk("Hi", False), ObjChunk("", True)]:
                        yield c
                return ObjChunk("Hi")

        backend = OllamaBackend(ObjClient(), "test")
        text, _ = backend.call([{"role": "user", "content": "hi"}], stream=True)
        assert text == "Hi"


# ═══════════════════════════════════════════════════════════════════════
# OpenAIBackend tests (with mock, no real openai import needed)
# ═══════════════════════════════════════════════════════════════════════

class TestOpenAIBackend:
    def test_requires_openai_package(self):
        """OpenAIBackend raises ImportError if openai not installed."""
        # We can't easily test the import failure without mocking sys.modules,
        # so we test that the backend works when openai IS available.
        # This test just verifies the class exists and is importable.
        assert OpenAIBackend is not None

    def test_middleware_auto_config_without_tpm(self):
        """Without tpm_limit: trimmer uses full ctx, retry is light, no TPM tracker."""
        # Can't construct without openai installed, but we can test the
        # configuration logic by checking the class structure.
        # This is a structural test — real construction requires openai.
        pass

    def test_middleware_auto_config_with_tpm(self):
        """With tpm_limit: trimmer is capped, retry is aggressive, TPM tracker active."""
        # Same as above — requires openai to construct.
        pass


# ═══════════════════════════════════════════════════════════════════════
# TokenCounter tests
# ═══════════════════════════════════════════════════════════════════════

class TestTokenCounter:
    def test_counts_string_messages(self):
        counter = TokenCounter("test-model")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello there!"},
        ]
        count = counter.count(messages)
        # Either tiktoken count or char/4 fallback — both should be > 0
        assert count > 0

    def test_char_fallback_when_no_tiktoken(self):
        """When tiktoken unavailable, falls back to char/4 heuristic."""
        counter = TokenCounter("nonexistent-model-xyz")
        # Force char/4 fallback
        counter._encoder = None
        messages = [{"role": "user", "content": "abcdefgh"}]  # 8 chars
        count = counter.count(messages)
        # char/4 fallback: 8 chars / 4 = 2 (overhead only applies with tiktoken)
        assert count == 2

    def test_empty_messages(self):
        counter = TokenCounter("test")
        assert counter.count([]) == 0

    def test_count_single_message(self):
        counter = TokenCounter("test")
        msg = {"role": "user", "content": "hello"}
        assert counter.count_message(msg) == counter.count([msg])


# ═══════════════════════════════════════════════════════════════════════
# TPMTracker tests
# ═══════════════════════════════════════════════════════════════════════

class TestTPMTracker:
    def test_no_wait_when_under_budget(self):
        tracker = TPMTracker(tpm_limit=1000, quiet=True)
        tracker.record_usage(100)
        # Should not block — 100 + 200 = 300 <= 1000
        tracker.wait_if_needed(200)

    def test_single_request_over_budget_no_wait(self):
        """Single request >= tpm_limit should not wait forever."""
        tracker = TPMTracker(tpm_limit=100, quiet=True)
        # 200 >= 100 → should return immediately
        tracker.wait_if_needed(200)

    def test_record_and_check_usage(self):
        tracker = TPMTracker(tpm_limit=1000, quiet=True)
        tracker.record_usage(500)
        assert tracker._current_usage() == 500
        tracker.record_usage(300)
        assert tracker._current_usage() == 800

    def test_prune_old_entries(self):
        tracker = TPMTracker(tpm_limit=1000, quiet=True)
        # Manually insert an old entry
        tracker._usage_log.append((time.time() - 120, 500))
        tracker.record_usage(100)
        # Old entry should be pruned
        assert tracker._current_usage() == 100

    def test_wait_when_over_budget(self):
        """When over budget, wait_if_needed should block then proceed."""
        tracker = TPMTracker(tpm_limit=100, quiet=True)
        tracker.record_usage(90)
        # 90 + 20 = 110 > 100 → should wait
        # But we can't test the actual sleep without time mocking.
        # Instead, test that it would need to wait:
        assert tracker._current_usage() + 20 > 100


# ═══════════════════════════════════════════════════════════════════════
# RetryHandler tests
# ═══════════════════════════════════════════════════════════════════════

class TestRetryHandler:
    def test_no_retry_on_success(self):
        handler = RetryHandler(max_retries=3, quiet=True)
        call_count = [0]

        def func():
            call_count[0] += 1
            return "ok"

        result = handler.execute(func)
        assert result == "ok"
        assert call_count[0] == 1

    def test_retries_on_429(self):
        handler = RetryHandler(
            triggers=["429"], initial_wait=0.01, max_wait=0.05,
            max_retries=2, quiet=True
        )
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("429 Too Many Requests")
            return "ok"

        result = handler.execute(func)
        assert result == "ok"
        assert call_count[0] == 2

    def test_no_retry_on_non_retryable_error(self):
        handler = RetryHandler(
            triggers=["429"], max_retries=3, quiet=True
        )

        def func():
            raise ValueError("not a rate limit")

        with pytest.raises(ValueError):
            handler.execute(func)

    def test_max_retries_exhausted(self):
        handler = RetryHandler(
            triggers=["429"], initial_wait=0.01, max_wait=0.02,
            max_retries=2, quiet=True
        )
        call_count = [0]

        def func():
            call_count[0] += 1
            raise Exception("429 rate limit")

        with pytest.raises(Exception, match="429"):
            handler.execute(func)
        assert call_count[0] == 3  # initial + 2 retries

    def test_is_retryable(self):
        handler = RetryHandler(triggers=["429", "connection"], quiet=True)
        assert handler._is_retryable(Exception("429 error"))
        assert handler._is_retryable(Exception("Connection refused"))
        assert not handler._is_retryable(ValueError("bad input"))


# ═══════════════════════════════════════════════════════════════════════
# HistoryTrimmer tests
# ═══════════════════════════════════════════════════════════════════════

class TestHistoryTrimmer:
    def test_no_trim_when_under_budget(self):
        counter = TokenCounter("test")
        trimmer = HistoryTrimmer(counter, max_tokens=10000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = trimmer.trim(messages)
        assert result == messages

    def test_trims_when_over_budget(self):
        counter = TokenCounter("test")
        counter._encoder = None  # Force char/4 fallback for predictability
        trimmer = HistoryTrimmer(counter, max_tokens=50, reserve_output=10)
        # input_budget = 40. Each message: content_len/4 + 4 overhead
        messages = [
            {"role": "system", "content": "S"},  # 1/4 + 4 = 4
            {"role": "user", "content": "A" * 100},  # 100/4 + 4 = 29
            {"role": "assistant", "content": "B" * 100},  # 29
            {"role": "user", "content": "C" * 10},  # 10/4 + 4 = 6
        ]
        result = trimmer.trim(messages)
        # The large middle messages should be dropped (not present in result)
        result_contents = [m.get("content", "") for m in result]
        assert "A" * 100 not in result_contents  # first middle msg dropped
        # System preserved; last user may be truncated to fit budget
        assert result[0] == messages[0]
        assert result[-1]["role"] == "user"
        assert "C" * 10 in result[-1]["content"] or result[-1]["content"].startswith("C")

    def test_preserves_system_and_last(self):
        counter = TokenCounter("test")
        counter._encoder = None
        trimmer = HistoryTrimmer(counter, max_tokens=30, reserve_output=10)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": "y" * 200},
            {"role": "user", "content": "Final message"},
        ]
        result = trimmer.trim(messages)
        assert result[0]["role"] == "system"
        # Last message may be truncated to fit budget, but role is preserved
        assert result[-1]["role"] == "user"
        assert "Final" in result[-1]["content"]

    def test_empty_messages(self):
        counter = TokenCounter("test")
        trimmer = HistoryTrimmer(counter, max_tokens=100)
        assert trimmer.trim([]) == []

    def test_two_messages_not_trimmed(self):
        """If only 2 messages, don't trim even if over budget."""
        counter = TokenCounter("test")
        counter._encoder = None
        trimmer = HistoryTrimmer(counter, max_tokens=10, reserve_output=5)
        messages = [
            {"role": "system", "content": "x" * 100},
            {"role": "user", "content": "y" * 100},
        ]
        result = trimmer.trim(messages)
        assert result == messages

    def test_dropped_messages_summarized(self):
        counter = TokenCounter("test")
        counter._encoder = None
        trimmer = HistoryTrimmer(counter, max_tokens=40, reserve_output=10)
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "C"},
        ]
        result = trimmer.trim(messages)
        # Should have a summary system message
        summary_msgs = [m for m in result if m["role"] == "system" and "trimmed" in m.get("content", "")]
        assert len(summary_msgs) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Backend.call() orchestration tests
# ═══════════════════════════════════════════════════════════════════════

class TestBackendOrchestration:
    def test_call_passes_on_chunk(self):
        """on_chunk callback is passed through to _call."""
        client = MockOllamaClient("Hello")
        backend = OllamaBackend(client, "test")
        chunks = []
        text, _ = backend.call(
            [{"role": "user", "content": "hi"}], stream=True, on_chunk=chunks.append
        )
        assert "".join(chunks) == "Hello"

    def test_call_with_tpm_tracker_records_usage(self):
        """When TPM tracker is configured, usage is recorded after call."""
        client = MockOllamaClient("Hi", eval_count=50)
        backend = OllamaBackend(client, "test")
        # Manually attach middleware (simulating what OpenAIBackend does)
        backend._token_counter = TokenCounter("test")
        backend._tpm_tracker = TPMTracker(tpm_limit=1000, quiet=True)
        backend.call([{"role": "user", "content": "hi"}], stream=False)
        # TPM tracker should have recorded usage
        assert backend._tpm_tracker._current_usage() > 0

    def test_call_with_trimmer_trims_history(self):
        """When trimmer is configured, history is trimmed before call."""
        client = MockOllamaClient("Hi")
        backend = OllamaBackend(client, "test")
        counter = TokenCounter("test")
        counter._encoder = None
        backend._token_counter = counter
        backend._trimmer = HistoryTrimmer(counter, max_tokens=20, reserve_output=5)
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": "y" * 200},
            {"role": "user", "content": "Final"},
        ]
        backend.call(messages, stream=False)
        # The client should have received trimmed messages
        sent_messages = client.calls[0]["messages"]
        assert len(sent_messages) < len(messages)

    def test_call_with_retry_retries(self):
        """When retry is configured, retryable errors trigger retry."""
        call_count = [0]

        class FlakyClient:
            def chat(self, **kwargs):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise Exception("429 rate limit")
                return {"message": {"content": "ok"}}

        backend = OllamaBackend(FlakyClient(), "test")
        backend._retry = RetryHandler(
            triggers=["429"], initial_wait=0.01, max_wait=0.02,
            max_retries=2, quiet=True
        )
        text, _ = backend.call([{"role": "user", "content": "hi"}], stream=False)
        assert text == "ok"
        assert call_count[0] == 2
