"""AgenticSession: the feedback loop that ties parser, actions, and model together."""

import json
import sys
from datetime import date

from .constants import (
    MODEL_TEMPERATURE, MAX_FEEDBACK_ROUNDS,
    MAX_IDENTICAL_ACTION_REPEATS, get_platform_info,
)
from .parser import parse_actions, extract_text_outside_blocks
from .actions import ActionExecutor, agent_print
from .backends import OllamaBackend, Backend
from .system_prompt import build_system_prompt
from .exceptions import AgenticInterrupted


class AgenticSession:
    """Interactive agentic chat session with an Ollama-compatible LLM.

    The session runs a feedback loop:
        1. Send conversation to model, get response text.
        2. Parse action blocks (WRITE/EDIT/FILE/RUN/TOOL/SKILL) from response.
        3. Execute actions, collect observations.
        4. If actions were found, feed observations back as a user message
           and repeat from step 1 (up to MAX_FEEDBACK_ROUNDS).
        5. If no actions, the response is final.

    Args:
        client: Any object with a .chat(model=, messages=, stream=) method
                matching the Ollama API. The core does not import the ollama
                library directly.
        model: Model name string (e.g. "glm-5.1:cloud").
        workdir: Working directory for file operations.
        tools: Enable TOOL: protocol and built-in tools.
        skills: Enable SKILL: protocol.
        system_prompt: Custom system prompt string. If None, built from defaults.
        name: Agent name for the default system prompt.
        confirm_callback: Function(action_type, details) -> bool for confirmation.
        auto_approve_safe: Auto-approve safe shell commands.
        stream: Whether to stream model responses.
        temperature: Model temperature.
        on_chunk: Callback(chunk_text) called during streaming for real-time display.
    """

    def __init__(self, client=None, model=None, workdir=".", tools=True, skills=False,
                 system_prompt=None, name="uhu", confirm_callback=None,
                 auto_approve_safe=False, stream=True, temperature=None,
                 max_feedback_rounds=None, config_filename="agentic-core.json",
                 on_chunk=None, backend=None):
        self.temperature = temperature if temperature is not None else MODEL_TEMPERATURE

        # Backend: explicit backend, or auto-wrap client in OllamaBackend
        if backend is not None:
            self.backend = backend
        elif client is not None:
            self.backend = OllamaBackend(client, model, self.temperature)
        else:
            raise ValueError("Either client= or backend= is required")

        self.client = client  # kept for backward compat
        self.model = model
        self.workdir = workdir
        self.tools = tools
        self.skills = skills
        self.stream = stream
        self.on_chunk = on_chunk
        self.max_feedback_rounds = max_feedback_rounds or MAX_FEEDBACK_ROUNDS

        # Build system prompt
        if system_prompt is None:
            extra = []
            if tools:
                from .tools import tools_system_prompt
                tp = tools_system_prompt()
                if tp:
                    extra.append(tp)
            if skills:
                from .skills import skills_system_prompt
                sp = skills_system_prompt()
                if sp:
                    extra.append(sp)
            system_prompt = build_system_prompt(
                name=name, tools=tools, skills=skills, extra_sections=extra,
            )
        self.system_prompt = system_prompt

        # Conversation history
        self.history = [{"role": "system", "content": system_prompt}]

        # Action executor
        self.executor = ActionExecutor(
            workdir=workdir,
            confirm_callback=confirm_callback,
            auto_approve_safe=auto_approve_safe,
            config_filename=config_filename,
        )

        # Loop detection state
        self._last_action_signature = None
        self._repeat_count = 0

        # Native tool_calls snapshot from the last model call — consumed
        # exactly once per action round by run() (never leaks between rounds).
        self._pending_native = None

    def run(self, user_text):
        """Send user text and run the agentic feedback loop.

        Returns the final model response text (with action blocks stripped).

        Raises AgenticInterrupted if the user presses Ctrl+C. The conversation
        history is left in a consistent state: the user message and any
        partial assistant response / observations are saved before raising.
        """
        self.history.append({"role": "user", "content": user_text})

        try:
            final_text = ""
            for round_num in range(self.max_feedback_rounds + 1):
                # Call the model
                response_text = self._call_model()

                # Parse actions, then merge any native tool_calls the backend
                # captured (native-calling models express actions via the
                # structured tool_calls channel instead of text blocks).
                pending = self._take_pending_native()
                actions = parse_actions(response_text)
                native_actions = self._native_to_actions(pending)
                if native_actions:
                    actions = (actions or []) + native_actions

                if not actions:
                    # No actions — this is the final response
                    self.history.append({"role": "assistant", "content": response_text})
                    final_text = response_text
                    break

                # Check for repeated identical actions (loop detection)
                sig = self._action_signature(actions)
                if sig == self._last_action_signature:
                    self._repeat_count += 1
                    if self._repeat_count >= MAX_IDENTICAL_ACTION_REPEATS:
                        agent_print(f"[Loop detected: identical actions repeated "
                                    f"{self._repeat_count} times. Stopping.]")
                        self.history.append({"role": "assistant", "content": response_text})
                        final_text = response_text
                        break
                else:
                    self._last_action_signature = sig
                    self._repeat_count = 0

                # Store assistant response; raw native tool_calls stay attached
                # so tool-role results can immediately follow on the wire
                # (some servers 400 when other roles interleave).
                entry = {"role": "assistant", "content": response_text}
                if pending and pending.get("raw"):
                    entry["tool_calls"] = pending["raw"]
                self.history.append(entry)

                # Execute actions
                observations = self.executor.execute_actions(actions)

                if not observations:
                    # No observations (all skipped) — nudge the model
                    observations = ["[All actions were skipped. Please try a different approach.]"]

                # Feed observations back: per-call tool-role messages when the
                # round used native tool_calls, else a single user message.
                self._append_observations(observations, len(native_actions))

                final_text = response_text

                if round_num == self.max_feedback_rounds:
                    agent_print(f"[Max feedback rounds ({self.max_feedback_rounds}) reached.]")

            # Return prose without action blocks
            return extract_text_outside_blocks(final_text)

        except AgenticInterrupted as e:
            # Fix history: save partial assistant response if we have one
            if e.phase == "streaming":
                # Interrupted during model call — save partial text as assistant
                if e.partial_text:
                    self.history.append({"role": "assistant", "content": e.partial_text})
                else:
                    # No partial text — remove the user message we added
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
            elif e.phase in ("actions", "confirmation"):
                # Interrupted during action execution — save observations as user msg
                if e.partial_text:
                    self.history.append({"role": "user", "content": e.partial_text})
            # Re-raise so the app can handle UX (print message, continue loop)
            raise

    def _call_model(self):
        """Call the model and return the full response text.

        Also snapshots native tool_calls (if the backend captured any) into
        self._pending_native for the caller to merge into parsed actions."""
        if self.stream:
            text = self._call_model_stream()
        else:
            text = self._call_model_nonstream()
        raw = [dict(c) for c in (getattr(self.backend, "last_raw_tool_calls", None) or [])]
        norm = list(getattr(self.backend, "last_tool_calls", None) or [])
        self._pending_native = {"raw": raw, "norm": norm} if norm else None
        if norm and not (text or "").strip():
            # Model produced only structured tool_calls — render them for
            # visibility (streaming showed nothing on screen).
            self._render_native_display(norm)
        return text

    def _take_pending_native(self):
        """Return (and clear) the pending native tool_calls snapshot."""
        pending = getattr(self, "_pending_native", None)
        self._pending_native = None
        return pending

    @staticmethod
    def _native_to_actions(pending):
        """Convert normalized native tool_calls to executor action dicts."""
        if not pending:
            return []
        return [{"type": "tool", "path": tc["name"], "code": "",
                 "params": tc.get("params", {})}
                for tc in pending.get("norm", [])]

    @staticmethod
    def _render_native_display(norm_calls):
        """Print a compact display for native tool_calls with no text content."""
        for i, tc in enumerate(norm_calls, 1):
            try:
                args_json = json.dumps(tc.get("params", {}), default=str)
            except (TypeError, ValueError):
                args_json = str(tc.get("params", {}))
            preview = args_json[:400] + ("..." if len(args_json) > 400 else "")
            agent_print(f"[native tool_call {i}/{len(norm_calls)}: {tc['name']}]\n"
                        f"{preview}\n")

    def _append_observations(self, observations, native_count=0):
        """Append observations to history.

        With native tool_calls this round, the observation for each call is
        attached as a tool-role message immediately after the assistant
        entry (wire requirement). Native actions are appended last, so their
        observations are the last entries; any remaining observations
        (text-block actions) go back as a single user message."""
        last = self.history[-1] if self.history else None
        if native_count and isinstance(last, dict) and last.get("tool_calls"):
            if len(observations) >= native_count:
                native_obs = observations[-native_count:]
                rest = observations[:-native_count]
            else:
                native_obs = (["[skipped]"] * (native_count - len(observations))
                              + observations)
                rest = []
            for i, call in enumerate(last["tool_calls"]):
                msg = {"role": "tool",
                       "content": native_obs[i] if i < len(native_obs) else "[skipped]",
                       "name": (call.get("function") or {}).get("name", "")}
                if call.get("id"):
                    msg["tool_call_id"] = call["id"]
                self.history.append(msg)
            if rest:
                self.history.append({"role": "user", "content": "\n\n".join(rest)})
            return
        obs_text = "\n\n".join(observations)
        self.history.append({"role": "user", "content": obs_text})

    def _call_model_stream(self):
        """Stream the model response, calling on_chunk for each piece.

        Raises AgenticInterrupted on Ctrl+C with partial text collected so far.
        """
        chunks = []

        def collector(text):
            chunks.append(text)
            if self.on_chunk:
                self.on_chunk(text)
            else:
                sys.stdout.write(text)
                sys.stdout.flush()

        try:
            text, _ = self.backend.call(
                self.history, stream=True, on_chunk=collector
            )
            if not self.on_chunk:
                print()  # newline after streaming
            return text
        except KeyboardInterrupt:
            partial = "".join(chunks)
            if not self.on_chunk:
                print("\n[interrupted]")
            raise AgenticInterrupted(partial, phase="streaming")
        except Exception as e:
            error_msg = f"[Model error: {e}]"
            if self.on_chunk:
                self.on_chunk(error_msg)
            else:
                print(error_msg)
            return error_msg

    def _call_model_nonstream(self):
        """Get the full model response without streaming.

        Raises AgenticInterrupted on Ctrl+C.
        """
        try:
            text, _ = self.backend.call(self.history, stream=False)
            return text
        except KeyboardInterrupt:
            raise AgenticInterrupted("", phase="streaming")
        except Exception as e:
            return f"[Model error: {e}]"

    @staticmethod
    def _action_signature(actions):
        """Create a hashable signature of actions for loop detection.

        Includes a params hash so the same tool called with different
        args is NOT misdetected as a looping action (upstream 65c5579)."""
        import json as _json
        parts = []
        for a in actions:
            try:
                params_hash = _json.dumps(a.get("params", {}),
                                          sort_keys=True, default=str)[:200]
            except (TypeError, ValueError):
                params_hash = str(a.get("params", ""))[:200]
            parts.append(f"{a['type']}:{a.get('path', '')}:"
                         f"{a.get('code', '')[:200]}:{params_hash}")
        return "|".join(parts)

    def reset(self):
        """Clear conversation history, keeping only the system prompt."""
        self.history = [self.history[0]]
        self._last_action_signature = None
        self._repeat_count = 0
        self._pending_native = None
