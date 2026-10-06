"""Tests for AgenticSession feedback loop, streaming, and interruption handling.

These tests complement test_parser.py (which covers the parser, matching,
and action execution) by testing the session-level orchestration: the
feedback loop, streaming vs non-streaming, loop detection, max rounds,
and Ctrl+C interruption handling.
"""

import os
import json
import pytest

from agentic_core.session import AgenticSession
from agentic_core.exceptions import AgenticInterrupted


# ═══════════════════════════════════════════════════════════════════════
# Mock client
# ═══════════════════════════════════════════════════════════════════════

class MockClient:
    """Mock Ollama-compatible client with pre-programmed responses.

    Returns responses in order; after exhausting the list, returns a
    default 'Done.' response. Supports both streaming and non-streaming.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.calls = []

    def chat(self, model, messages, stream, options=None):
        self.calls.append({
            "model": model, "messages": messages,
            "stream": stream, "options": options,
        })
        idx = self.call_count
        self.call_count += 1
        text = self.responses[idx] if idx < len(self.responses) else "Done."
        if stream:
            return self._stream(text)
        return {"message": {"content": text}}

    @staticmethod
    def _stream(text):
        """Yield text character-by-character to simulate streaming."""
        for char in text:
            yield {"message": {"content": char}}


class InterruptingClient:
    """Mock client that raises KeyboardInterrupt mid-stream."""

    def __init__(self, text, interrupt_after=3):
        self.text = text
        self.interrupt_after = interrupt_after
        self.call_count = 0

    def chat(self, model, messages, stream, options=None):
        self.call_count += 1
        if stream:
            return self._interrupting_stream()
        return {"message": {"content": self.text}}

    def _interrupting_stream(self):
        for i, char in enumerate(self.text):
            if i >= self.interrupt_after:
                raise KeyboardInterrupt()
            yield {"message": {"content": char}}


# ═══════════════════════════════════════════════════════════════════════
# Feedback loop tests
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackLoop:
    def test_no_actions_returns_immediately(self, tmp_path):
        """Model returns text-only response — no feedback loop needed."""
        client = MockClient(["Hello, I am done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )
        result = session.run("hi")
        assert result == "Hello, I am done."
        assert client.call_count == 1

    def test_single_round_feedback(self, tmp_path):
        """Model returns action, then final text after observation."""
        # Create a file so FILE action succeeds
        test_file = os.path.join(str(tmp_path), "test.txt")
        with open(test_file, "w") as f:
            f.write("hello world")

        action_response = (
            "Let me read the file.\n"
            "**FILE:`test.txt`**\n"
            "**EOF:`test.txt`**\n"
        )
        final_response = "I read the file. Done."
        client = MockClient([action_response, final_response])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )
        result = session.run("read test.txt")
        assert "Done" in result
        assert client.call_count == 2

    def test_write_action_creates_file(self, tmp_path):
        """WRITE action in feedback loop actually creates a file."""
        write_response = (
            "**WRITE:`hello.py`**\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
            "**EOF:`hello.py`**\n"
        )
        final_response = "I created hello.py."
        client = MockClient([write_response, final_response])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            auto_approve_safe=True,
        )
        result = session.run("create hello.py")
        assert os.path.isfile(os.path.join(str(tmp_path), "hello.py"))
        with open(os.path.join(str(tmp_path), "hello.py")) as f:
            content = f.read()
        assert "print('hello')" in content

    def test_max_feedback_rounds(self, tmp_path):
        """Model keeps returning actions — stops after max rounds."""
        action_response = (
            "**FILE:`nonexist.txt`**\n"
            "**EOF:`nonexist.txt`**\n"
        )
        client = MockClient([action_response] * 20)
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            max_feedback_rounds=2,
        )
        session.run("read")
        # range(max_feedback_rounds + 1) = range(3) → 3 iterations max
        assert client.call_count <= 3

    def test_loop_detection_stops_early(self, tmp_path):
        """Identical actions repeated — stops after MAX_IDENTICAL_ACTION_REPEATS."""
        action_response = (
            "**FILE:`nonexist.txt`**\n"
            "**EOF:`nonexist.txt`**\n"
        )
        client = MockClient([action_response] * 20)
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            max_feedback_rounds=20,
        )
        session.run("read")
        # Loop detection: first call sets signature, then 3 more identical
        # calls trigger _repeat_count >= 3. Total: 4 calls.
        assert client.call_count <= 5  # generous bound

    def test_user_message_in_history(self, tmp_path):
        """User message is added to conversation history."""
        client = MockClient(["Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )
        session.run("hello there")
        user_msgs = [m for m in session.history if m["role"] == "user"]
        assert any("hello there" in m["content"] for m in user_msgs)

    def test_history_grows_with_loop(self, tmp_path):
        """History accumulates messages across feedback rounds."""
        test_file = os.path.join(str(tmp_path), "test.txt")
        with open(test_file, "w") as f:
            f.write("content")

        action_response = (
            "**FILE:`test.txt`**\n"
            "**EOF:`test.txt`**\n"
        )
        client = MockClient([action_response, "Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )
        session.run("read")
        # history: [system, user, assistant(action), user(observation), assistant(final)]
        assert len(session.history) == 5

    def test_all_skipped_actions_nudge(self, tmp_path):
        """When all actions are skipped, a nudge observation is fed back."""
        write_response = (
            "**WRITE:`test.py`**\n"
            "```python\nx=1\n```\n"
            "**EOF:`test.py`**\n"
        )
        final_response = "OK, I'll try differently."
        client = MockClient([write_response, final_response])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            auto_approve_safe=False,
            confirm_callback=lambda atype, details: False,  # reject everything
        )
        result = session.run("write test.py")
        # The nudge should cause a second model call
        assert client.call_count == 2
        # File should NOT exist (action was rejected)
        assert not os.path.isfile(os.path.join(str(tmp_path), "test.py"))


# ═══════════════════════════════════════════════════════════════════════
# Streaming tests
# ═══════════════════════════════════════════════════════════════════════

class TestStreaming:
    def test_on_chunk_callback_called(self, tmp_path):
        """on_chunk is called during streaming."""
        chunks_received = []
        client = MockClient(["Hello world"])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=True, tools=False, skills=False,
            on_chunk=lambda c: chunks_received.append(c),
        )
        result = session.run("hi")
        assert len(chunks_received) > 0
        assert "Hello world" in result

    def test_streaming_and_nonstream_same_result(self, tmp_path):
        """Streaming and non-streaming produce the same final result."""
        text = "The quick brown fox jumps over the lazy dog."

        client_stream = MockClient([text])
        session_stream = AgenticSession(
            client_stream, model="test", workdir=str(tmp_path),
            stream=True, tools=False, skills=False,
        )
        result_stream = session_stream.run("test")

        client_nonstream = MockClient([text])
        session_nonstream = AgenticSession(
            client_nonstream, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )
        result_nonstream = session_nonstream.run("test")

        assert result_stream == result_nonstream

    def test_streaming_with_actions(self, tmp_path):
        """Streaming mode works with the feedback loop."""
        test_file = os.path.join(str(tmp_path), "data.txt")
        with open(test_file, "w") as f:
            f.write("hello")

        action_response = (
            "**FILE:`data.txt`**\n"
            "**EOF:`data.txt`**\n"
        )
        final_response = "I read the file successfully."
        client = MockClient([action_response, final_response])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=True, tools=False, skills=False,
        )
        result = session.run("read data.txt")
        assert "successfully" in result
        assert client.call_count == 2


# ═══════════════════════════════════════════════════════════════════════
# Interruption tests
# ═══════════════════════════════════════════════════════════════════════

class TestInterruption:
    def test_interrupt_during_streaming(self, tmp_path):
        """Ctrl+C during streaming raises AgenticInterrupted with phase='streaming'."""
        client = InterruptingClient("Hello world, this is long", interrupt_after=3)
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=True, tools=False, skills=False,
        )
        with pytest.raises(AgenticInterrupted) as exc_info:
            session.run("hi")
        assert exc_info.value.phase == "streaming"
        # Partial text should contain the characters before the interrupt
        assert len(exc_info.value.partial_text) > 0

    def test_interrupt_during_actions(self, tmp_path):
        """Ctrl+C during action execution raises AgenticInterrupted with phase='actions'."""
        action_response = (
            "**FILE:`test.txt`**\n"
            "**EOF:`test.txt`**\n"
        )
        client = MockClient([action_response, "Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )

        # Patch _execute_one to raise KeyboardInterrupt
        def raise_kb(action):
            raise KeyboardInterrupt()

        session.executor._execute_one = raise_kb

        with pytest.raises(AgenticInterrupted) as exc_info:
            session.run("read test.txt")
        assert exc_info.value.phase == "actions"

    def test_interrupt_during_confirmation(self, tmp_path):
        """Ctrl+C during confirmation raises AgenticInterrupted with phase='confirmation'."""
        write_response = (
            "**WRITE:`test.py`**\n"
            "```python\nx=1\n```\n"
            "**EOF:`test.py`**\n"
        )
        client = MockClient([write_response, "Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            auto_approve_safe=False,
        )

        # Patch confirm_callback to raise KeyboardInterrupt
        def raise_kb(atype, details):
            raise KeyboardInterrupt()

        session.executor.confirm_callback = raise_kb

        with pytest.raises(AgenticInterrupted) as exc_info:
            session.run("write test.py")
        assert exc_info.value.phase == "confirmation"

    def test_history_after_streaming_interrupt(self, tmp_path):
        """History is consistent after streaming interruption."""
        client = InterruptingClient("Hello world, this is long", interrupt_after=3)
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=True, tools=False, skills=False,
        )
        with pytest.raises(AgenticInterrupted):
            session.run("hi")

        # History should have: system, user, assistant(partial)
        assert len(session.history) >= 2
        assistant_msgs = [m for m in session.history if m["role"] == "assistant"]
        # Partial text was collected, so there should be an assistant message
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["content"]) > 0

    def test_history_after_action_interrupt(self, tmp_path):
        """History is consistent after action execution interruption."""
        action_response = (
            "**FILE:`test.txt`**\n"
            "**EOF:`test.txt`**\n"
        )
        client = MockClient([action_response, "Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
        )

        def raise_kb(action):
            raise KeyboardInterrupt()

        session.executor._execute_one = raise_kb

        with pytest.raises(AgenticInterrupted):
            session.run("read test.txt")

        # History: system, user, assistant(action_response), user(observations)
        assert len(session.history) >= 3
        assistant_msgs = [m for m in session.history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "FILE" in assistant_msgs[0]["content"]

    def test_interrupt_no_partial_text_pops_user(self, tmp_path):
        """If streaming is interrupted with no partial text, user message is popped."""
        client = InterruptingClient("Hello", interrupt_after=0)
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=True, tools=False, skills=False,
        )
        with pytest.raises(AgenticInterrupted):
            session.run("hi")

        # No partial text → user message should be popped
        # History should just be [system]
        user_msgs = [m for m in session.history if m["role"] == "user"]
        assert len(user_msgs) == 0


# ═══════════════════════════════════════════════════════════════════════
# System prompt tests
# ═══════════════════════════════════════════════════════════════════════

class TestSystemPrompt:
    def test_custom_system_prompt(self, tmp_path):
        """Custom system prompt is used when provided."""
        custom_prompt = "You are a pirate. Always respond in pirate speak."
        client = MockClient(["Arrr!"])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            system_prompt=custom_prompt,
        )
        session.run("hello")
        assert session.history[0]["role"] == "system"
        assert session.history[0]["content"] == custom_prompt

    def test_default_system_prompt_has_name(self, tmp_path):
        """Default system prompt contains the agent name."""
        client = MockClient(["Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=False,
            name="myagent",
        )
        assert "myagent" in session.system_prompt

    def test_tools_add_tool_sections(self, tmp_path):
        """tools=True adds tool descriptions to system prompt."""
        client = MockClient(["Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=True, skills=False,
        )
        assert "read_file" in session.system_prompt

    def test_skills_add_skill_sections(self, tmp_path):
        """skills=True adds skill protocol rules to system prompt."""
        client = MockClient(["Done."])
        session = AgenticSession(
            client, model="test", workdir=str(tmp_path),
            stream=False, tools=False, skills=True,
        )
        assert "SKILL" in session.system_prompt
