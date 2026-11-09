"""Tests for native tool_calls support: normalizer, fragment merge,
backend capture, and session integration."""

import json
import unittest

from agentic_core.tool_calls import (
    normalize_tool_calls,
    merge_tool_call_fragments,
    _clean_native_name,
)
from agentic_core.backends import OllamaBackend
from agentic_core.session import AgenticSession


class TestNormalize(unittest.TestCase):
    def test_ollama_dict_format(self):
        raw = [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
        self.assertEqual(normalize_tool_calls(raw),
                         [{"name": "read_file", "params": {"path": "a.py"}}])

    def test_openai_json_string_args(self):
        raw = [{"function": {"name": "run_command",
                             "arguments": json.dumps({"command": "ls"})}}]
        self.assertEqual(normalize_tool_calls(raw),
                         [{"name": "run_command", "params": {"command": "ls"}}])

    def test_bad_json_degrades_to_raw(self):
        raw = [{"function": {"name": "x", "arguments": "{not json"}}]
        self.assertEqual(normalize_tool_calls(raw),
                         [{"name": "x", "params": {"_raw": "{not json"}}])

    def test_flat_format(self):
        self.assertEqual(normalize_tool_calls([{"name": "t", "arguments": None}]),
                         [{"name": "t", "params": {}}])

    def test_no_name_skipped(self):
        self.assertEqual(normalize_tool_calls([{"function": {"arguments": {}}}]), [])

    def test_marker_wrapped_name_cleaned(self):
        raw = [{"function": {"name": "TOOL:`py_compile`", "arguments": {}}}]
        self.assertEqual(normalize_tool_calls(raw),
                         [{"name": "py_compile", "params": {}}])

    def test_structured_param_key_coercion(self):
        args = json.dumps({"replacements": [{"search": "a", "replace": "b"}]})
        out = normalize_tool_calls([{"function": {"name": "replace_in_file",
                                                  "arguments": args}}])
        self.assertEqual(out[0]["params"]["replacements"],
                         [{"search": "a", "replace": "b"}])

    def test_freeform_string_kept(self):
        args = {"path": "x.py", "content": "[1, 2]"}
        out = normalize_tool_calls([{"function": {"name": "write_file",
                                                  "arguments": args}}])
        self.assertEqual(out[0]["params"]["content"], "[1, 2]")


class TestCleanName(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(_clean_native_name("TOOL:`a`"), "a")
        self.assertEqual(_clean_native_name("SKILL: `b`"), "b")
        self.assertEqual(_clean_native_name(" plain "), "plain")
        self.assertEqual(_clean_native_name("a:b"), "a:b")  # not a marker


class TestMergeFragments(unittest.TestCase):
    def test_openai_indexed_string_fragments(self):
        frags = [
            {"index": 0, "function": {"name": "rea", "arguments": '{"pa'}},
            {"index": 0, "function": {"name": "d_file", "arguments": 'th": "x"}'}},
            {"index": 1, "function": {"name": "t", "arguments": "{}"}},
        ]
        merged = merge_tool_call_fragments(frags)
        out = normalize_tool_calls(merged)
        self.assertEqual([c["name"] for c in out], ["read_file", "t"])
        self.assertEqual(out[0]["params"], {"path": "x"})

    def test_native_no_index_named_starts_new(self):
        frags = [
            {"function": {"name": "a", "arguments": {"x": 1}}},
            {"function": {"name": "b", "arguments": {}}},
        ]
        out = normalize_tool_calls(merge_tool_call_fragments(frags))
        self.assertEqual([c["name"] for c in out], ["a", "b"])

    def test_native_nameless_continues_previous(self):
        frags = [
            {"function": {"name": "a", "arguments": {"x": 1}}},
            {"function": {"name": "", "arguments": {"y": 2}}},
        ]
        out = normalize_tool_calls(merge_tool_call_fragments(frags))
        self.assertEqual(out, [{"name": "a", "params": {"x": 1, "y": 2}}])

    def test_empty(self):
        self.assertEqual(merge_tool_call_fragments([]), [])
        self.assertEqual(merge_tool_call_fragments(None), [])


class FakeOllamaClient:
    """Duck-typed client delivering chunks (dict style) via .chat().

    Backed by ONE shared iterator: successive .chat() calls continue the
    same stream (realistic — each call yields the next chunk group)."""
    def __init__(self, chunks):
        self._it = iter(chunks)

    def chat(self, model, messages, stream, options):
        return self._it


class TestOllamaBackendCapture(unittest.TestCase):
    def test_streaming_capture(self):
        chunks = [
            {"message": {"role": "assistant", "content": "Thinking..."},
             "done": False},
            {"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"function": {
                             "name": "read_file",
                             "arguments": {"path": "a"}}}]},
             "done": True},
        ]
        backend = OllamaBackend(FakeOllamaClient(chunks), "m")
        text, _ = backend.call([{"role": "user", "content": "hi"}], stream=True)
        self.assertEqual(text, "Thinking...")
        self.assertEqual(backend.last_tool_calls,
                         [{"name": "read_file", "params": {"path": "a"}}])
        # Raw wire shape preserved for history round-trip
        self.assertEqual(backend.last_raw_tool_calls[0]
                         ["function"]["name"], "read_file")

    def test_no_tool_calls(self):
        chunks = [{"message": {"role": "assistant", "content": "hi"}, "done": True}]
        backend = OllamaBackend(FakeOllamaClient(chunks), "m")
        backend.call([{"role": "user", "content": "hi"}], stream=True)
        self.assertEqual(backend.last_tool_calls, [])

    def test_state_reset_between_calls(self):
        chunks1 = [{"message": {"content": "",
                                "tool_calls": [{"function": {"name": "a",
                                                             "arguments": {}}}]},
                    "done": True},
                   {"message": {"content": "", "role": "assistant"},
                    "done": False}]
        chunks2 = [{"message": {"role": "assistant", "content": "done"},
                    "done": True}]
        backend = OllamaBackend(FakeOllamaClient(chunks1 + chunks2), "m")
        backend.call([{"role": "user", "content": "1"}], stream=True)
        self.assertEqual(len(backend.last_tool_calls), 1)
        backend.call([{"role": "user", "content": "2"}], stream=True)
        self.assertEqual(backend.last_tool_calls, [])


class TestSessionIntegration(unittest.TestCase):
    def test_native_call_becomes_tool_action(self):
        class BlockingClient:
            def chat(self, model, messages, stream, options):
                if len(messages) == 2:  # system + first user msg
                    return {"message": {"content": "", "tool_calls": [
                                {"function": {"name": "read_file",
                                              "arguments": {
                                                  "path": "does-not-exist.txt"}}}]},
                            "prompt_eval_count": 5}
                return {"message": {"content": "All done."}, "prompt_eval_count": 9}

        backend = OllamaBackend(BlockingClient(), "m")
        session = AgenticSession(backend=backend, model="m", workdir=".",
                                 tools=False, stream=False,
                                 auto_approve_safe=True)
        final = session.run("do it")
        self.assertIn("All done", final)
        roles = [m["role"] for m in session.history]
        # system, user, assistant(tool_calls), tool result, assistant final
        self.assertEqual(roles, ["system", "user", "assistant", "tool",
                                 "assistant"])
        self.assertIn("tool_calls", session.history[2])
        self.assertEqual(
            session.history[3]["name"], "read_file")

    def test_reset_clears_pending(self):
        backend = OllamaBackend(FakeOllamaClient(
            [{"message": {"role": "assistant", "content": "x"}, "done": True}]), "m")
        session = AgenticSession(backend=backend, model="m", workdir=".",
                                 tools=False, stream=False)
        session._pending_native = {"raw": [], "norm": [{"name": "x", "params": {}}]}
        session.reset()
        self.assertIsNone(session._pending_native)


class TestChannelMixingSteer(unittest.TestCase):
    """Port of uhu 6313cfa: text-action marker detection must come BEFORE the
    long-name blob guard, and the sentinel must steer the model back to one
    channel instead of failing as 'Unknown tool'."""

    FIELD_BLOB = ('EDIT:`app.py`\n```search-replace\n<<<<<<< SEARCH\n'
                  'old code line here that is quite long indeed yes\n'
                  '=======\n>>>>>>> REPLACE\n```' + 'x' * 200)

    def test_marker_detected_before_long_blob_guard(self):
        out = normalize_tool_calls([{"function": {"name": self.FIELD_BLOB,
                                                  "arguments": {}}}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "_text_action_mixing")
        self.assertTrue(out[0]["params"]["_raw"].startswith("EDIT:`app.py`"))

    def test_short_marker_name_still_steers(self):
        out = normalize_tool_calls([{"function": {"name": "WRITE:`hello.py`",
                                                  "arguments": {}}}])
        self.assertEqual(out, [{"name": "_text_action_mixing",
                                "params": {"_raw": "WRITE:`hello.py`"}}])

    def test_long_blob_without_marker_still_malformed(self):
        out = normalize_tool_calls([{"function": {"name": "T" + "x" * 300,
                                                  "arguments": {}}}])
        self.assertEqual(out[0]["name"], "_malformed_native_call")

    def test_malformed_message_content(self):
        from agentic_core.tool_calls import looks_like_text_action
        for marker in ("WRITE:", "EDIT:`f`", "FILE:`f`", "RUN:\n```bash"):
            self.assertTrue(looks_like_text_action(marker))
        self.assertFalse(looks_like_text_action("plain_name"))
        self.assertFalse(looks_like_text_action("a:b"))

    def test_steering_message_via_session(self):
        class MixingClient:
            def chat(self, model, messages, stream, options):
                if len(messages) == 2:
                    return {"message": {"content": "", "tool_calls": [
                                {"function": {"name": "_text_action_mixing",
                                              "arguments": {
                                                  "_raw": "EDIT:`app.py` <<< ..."}}}]},
                            "prompt_eval_count": 5}
                return {"message": {"content": "Recovered."},
                        "prompt_eval_count": 9}

        backend = OllamaBackend(MixingClient(), "m")
        session = AgenticSession(backend=backend, model="m", workdir=".",
                                 tools=False, stream=False,
                                 auto_approve_safe=True)
        final = session.run("do it")
        self.assertIn("Recovered", final)
        obs = session.history[3]["content"]
        self.assertIn("channel mixing", obs)
        self.assertIn("Option A", obs)
        self.assertIn("Option B", obs)
        self.assertIn("'replace_in_file'", obs)


if __name__ == "__main__":
    unittest.main()
