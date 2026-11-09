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
from agentic_core.actions import ActionExecutor


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


class TestToolsSchemas(unittest.TestCase):
    """Native solicitation: schemas built from the flat parameters dicts."""

    def test_all_builtin_tools_have_schemas(self):
        from agentic_core.tools import tools_schemas
        schemas = tools_schemas()
        names = [s["function"]["name"] for s in schemas]
        self.assertEqual(len(names), 16)
        self.assertIn("read_file", names)
        self.assertIn("replace_in_file", names)

    def test_schema_shape(self):
        from agentic_core.tools import tools_schemas
        schemas = {s["function"]["name"]: s["function"]
                   for s in tools_schemas()}
        fn = schemas["write_file"]
        self.assertEqual(fn["name"], "write_file")
        self.assertTrue(fn["description"])
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertIn("path", fn["parameters"]["required"])
        self.assertIn("path", fn["parameters"]["properties"])

    def test_float_alias_guarded(self):
        from agentic_core.tools import _flat_params_to_schema
        s = _flat_params_to_schema("t", "d", {"x": {"type": "float"}})
        self.assertEqual(s["function"]["parameters"]["properties"]["x"]["type"],
                         "number")

    def test_prose_only_tools_skipped(self):
        from agentic_core.tools import _flat_params_to_schema, tools_schemas
        # A tool without parameters is excluded (stays text-reachable)
        class ProseTool:
            name = "prose_tool_x"
            description = "d"
            system_prompt = ""
            auto_approve = True
        import agentic_core.tools as tm
        old = dict(tm._registry)
        try:
            tm.register(ProseTool())
            names = [s["function"]["name"] for s in tools_schemas()]
            self.assertNotIn("prose_tool_x", names)
        finally:
            tm._registry.clear()
            tm._registry.update(old)

    def test_enabled_names_filter(self):
        from agentic_core.tools import tools_schemas
        schemas = tools_schemas(enabled_names={"read_file", "time_now"})
        self.assertEqual(
            sorted(s["function"]["name"] for s in schemas),
            ["read_file", "time_now"])


class TestNativePrompt(unittest.TestCase):
    def test_native_channel(self):
        from agentic_core.tools import tools_native_prompt
        p = tools_native_prompt("native")
        self.assertIn("native tool calls", p)
        self.assertIn("NOT native tools", p)
        self.assertNotIn("PREFER", p)

    def test_hybrid_channel_prefers_text(self):
        from agentic_core.tools import tools_native_prompt
        p = tools_native_prompt("hybrid")
        self.assertIn("PREFER text action blocks", p)
        self.assertIn("backup tool names", p)
        self.assertIn("replace_in_file", p)


class TestBackendToolsInjection(unittest.TestCase):
    """tools_schemas → wire payload on both backends; drop-on-reject."""

    def test_ollama_tools_param_sent(self):
        captured = {}

        class CapClient:
            def chat(self, model, messages, stream, options, **kw):
                captured.update(kw)
                return {"message": {"role": "assistant", "content": "ok"},
                        "prompt_eval_count": 1}

        schemas = [{"type": "function", "function": {
            "name": "read_file", "description": "d",
            "parameters": {"type": "object", "properties": {},
                           "required": []}}}]
        backend = OllamaBackend(CapClient(), "m", tools_schemas=schemas)
        backend.call([{"role": "user", "content": "hi"}], stream=False)
        self.assertEqual(captured.get("tools"), schemas)

    def test_ollama_no_schemas_no_tools_param(self):
        captured = {}

        class CapClient:
            def chat(self, model, messages, stream, options, **kw):
                captured.update(kw)
                return {"message": {"role": "assistant", "content": "ok"},
                        "prompt_eval_count": 1}

        backend = OllamaBackend(CapClient(), "m")
        backend.call([{"role": "user", "content": "hi"}], stream=False)
        self.assertNotIn("tools", captured)

    def test_ollama_narrow_client_falls_back(self):
        class NarrowClient:
            def chat(self, model, messages, stream, options):
                return {"message": {"role": "assistant", "content": "ok"},
                        "prompt_eval_count": 1}

        backend = OllamaBackend(NarrowClient(), "m",
                                tools_schemas=[{"type": "function"}])
        text, _ = backend.call([{"role": "user", "content": "hi"}],
                               stream=False)
        self.assertEqual(text, "ok")
        self.assertTrue(backend._tools_rejected)

    def test_wire_tool_results_user_observation_split(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {}}}]},
            {"role": "user", "content": "the observation"},
            {"role": "assistant", "content": "done"},
        ]
        out = OllamaBackend._wire_tool_results(msgs)
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["system", "user", "assistant", "tool",
                                 "assistant"])
        self.assertIn("tool_call_id", out[3])
        # assistant tool_calls gains the synthesized id
        self.assertEqual(out[3]["tool_call_id"],
                         out[2]["tool_calls"][0]["id"])

    def test_wire_defers_system_past_tool_results(self):
        msgs = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "a", "arguments": {}}}]},
            {"role": "system", "content": "radar warning"},
            {"role": "user", "content": "obs"},
        ]
        out = OllamaBackend._wire_tool_results(msgs)
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["assistant", "tool", "system"])

    def test_openai_tools_in_kwargs(self):
        try:
            from agentic_core.backends import OpenAIBackend
        except ImportError:
            self.skipTest("openai not installed")


class TestSessionNativeModes(unittest.TestCase):
    """Session-level mode resolution and prompt selection."""

    def _prompt_of(self, **kw):
        class SilentClient:
            def chat(self, model, messages, stream, options, **k):
                return {"message": {"role": "assistant",
                                    "content": "final"},
                        "prompt_eval_count": 1}
        defaults = dict(backend=None, model="m", workdir=".",
                        tools=True, stream=False)
        defaults.update(kw)
        if defaults["backend"] is None:
            defaults.pop("backend")
            defaults["client"] = SilentClient()
        session = AgenticSession(**defaults)
        return session

    def test_default_text_mode_passive(self):
        s = self._prompt_of()
        self.assertEqual(s.tools_mode, "text")
        self.assertFalse(s.native_tools)
        self.assertIsNone(getattr(s.backend, "tools_schemas", None))
        self.assertIn("**TOOL:", s.system_prompt)  # prose listing present

    def test_hybrid_mode_attaches_schemas_and_prompt(self):
        s = self._prompt_of(native_tools=True, tools_mode="hybrid")
        self.assertTrue(s.native_tools)
        schemas = s.backend.tools_schemas
        self.assertTrue(schemas)
        self.assertIn(s.backend, [s.backend])  # sanity
        self.assertIn("PREFER text action blocks", s.system_prompt)
        self.assertNotIn("Available tools:", s.system_prompt)

    def test_native_mode_compact_prompt(self):
        s = self._prompt_of(native_tools=True, tools_mode="native")
        self.assertIn("NOT native tools", s.system_prompt)
        self.assertNotIn("PREFER text action blocks", s.system_prompt)

    def test_text_mode_flag_overrides(self):
        s = self._prompt_of(native_tools=True, tools_mode="text")
        self.assertFalse(s.native_tools)
        self.assertIn("**TOOL:", s.system_prompt)

    def test_bad_mode_degrades_to_text(self):
        s = self._prompt_of(native_tools=True, tools_mode="bogus")
        self.assertEqual(s.tools_mode, "text")
        self.assertFalse(s.native_tools)


class TestMixingCounter(unittest.TestCase):
    """1st mixing → plain steering; 3rd → native-only suggestion appended."""

    def _executor_with_count(self, n):
        ex = ActionExecutor(workdir=".", auto_approve_safe=True)
        ex._mixing_count = n
        return ex

    def test_first_event_no_suggestion(self):
        ex = self._executor_with_count(0)
        obs = ex._do_tool({"type": "tool", "path": "_text_action_mixing",
                           "code": "", "params": {"_raw": "WRITE:`x`"}})
        self.assertIn("channel mixing", obs)
        self.assertNotIn("third channel-mixing event", obs)
        self.assertEqual(ex._mixing_count, 1)

    def test_third_event_suggests_native_only(self):
        ex = self._executor_with_count(2)
        obs = ex._do_tool({"type": "tool", "path": "_text_action_mixing",
                           "code": "", "params": {"_raw": "EDIT:`x`"}})
        self.assertIn("third channel-mixing event", obs)
        self.assertIn("replace_in_file", obs)


class TestNativeModesEndToEnd(unittest.TestCase):
    """All three tools_mode branches, end-to-end on a scripted native client.

    Round 1: model emits a native tool_call (function-calling behavior);
    Round 2: final prose. Verifies per-mode wire + prompt + history shape."""

    class NativeThenFinalClient:
        def __init__(self):
            self.metas = []

        def chat(self, model, messages, stream, options, **kw):
            self.metas.append({"kw": kw, "roles": [m["role"] for m in messages]})
            if len(messages) == 2:
                return {"message": {"content": "", "tool_calls": [
                            {"function": {"name": "time_now",
                                          "arguments": {}}}]},
                        "prompt_eval_count": 5}
            return {"message": {"content": "final prose"}, "prompt_eval_count": 9}

    def _run(self, tools_mode):
        class _Client(self.NativeThenFinalClient):
            pass
        client = _Client()
        session = AgenticSession(client=client, model="m", workdir=".",
                                 tools=True, skills=False, stream=False,
                                 auto_approve_safe=True,
                                 native_tools=True, tools_mode=tools_mode)
        final = session.run("time")
        return session, client, final

    def test_hybrid_wire_prompt_history(self):
        session, client, final = self._run("hybrid")
        self.assertIn("final prose", final)
        meta = client.metas[0]
        # schemas on the wire
        names = [s["function"]["name"] for s in meta["kw"]["tools"]]
        self.assertIn("time_now", names)
        self.assertGreaterEqual(len(names), 16)
        # hybrid prompt: text blocks preferred + backup names
        self.assertIn("PREFER text action blocks", session.system_prompt)
        self.assertNotIn("Available tools:", session.system_prompt)
        # native call executed, tool-role result follows assistant
        self.assertEqual(client.metas[1]["roles"],
                         ["system", "user", "assistant", "tool"])
        self.assertGreater(len(session.history[3]["content"]), 5)

    def test_native_wire_prompt_history(self):
        session, client, final = self._run("native")
        self.assertIn("final prose", final)
        meta = client.metas[0]
        self.assertIn("tools", meta["kw"])
        self.assertIn("NOT native tools", session.system_prompt)
        self.assertNotIn("PREFER text action blocks", session.system_prompt)
        self.assertEqual(client.metas[1]["roles"],
                         ["system", "user", "assistant", "tool"])

    def test_text_no_wire_schemas_full_prompt(self):
        session, client, final = self._run_text_only()
        self.assertIn("final prose", final)
        meta = client.metas[0]
        self.assertNotIn("tools", meta["kw"])   # no solicitation
        self.assertIn("Available tools:", session.system_prompt)  # prose docs
        # spontaneous native call STILL merges (passive capture, always on)
        self.assertEqual(client.metas[1]["roles"],
                         ["system", "user", "assistant", "tool"])

    def _run_text_only(self):
        class _Client(self.NativeThenFinalClient):
            pass
        client = _Client()
        # tools_mode default ("text") — native_tools ignored
        session = AgenticSession(client=client, model="m", workdir=".",
                                 tools=True, skills=False, stream=False,
                                 auto_approve_safe=True)
        final = session.run("time")
        return session, client, final

    def test_skills_registry_present_in_all_modes(self):
        # Skills docs ride in the prompt in native modes too (they are not
        # tools — no schemas) — the SKILL: block protocol stays available.
        from agentic_core.skills import Skill, register as register_skill

        class TSkill(Skill):
            name = "mode_e2e_skill"
            description = "d"
            system_prompt = "## mode_e2e_skill"

            def execute(self, params, workdir=None, session=None):
                return "ok"

        register_skill(TSkill())
        try:
            for mode in ("text", "hybrid", "native"):
                session, client, _ = self._run_mode(mode)
                self.assertIn("mode_e2e_skill", session.system_prompt)
        finally:
            import agentic_core.skills as sm
            sm._registry.pop("mode_e2e_skill", None)

    def _run_mode(self, mode):
        client = self.NativeThenFinalClient()
        session = AgenticSession(client=client, model="m", workdir=".",
                                 tools=True, skills=True, stream=False,
                                 auto_approve_safe=True,
                                 native_tools=(mode != "text"),
                                 tools_mode=mode)
        session.run("time")
        return session, client, "x"
