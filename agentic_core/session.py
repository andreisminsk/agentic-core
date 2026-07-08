"""AgenticSession: the feedback loop that ties parser, actions, and model together."""

import sys
from datetime import date

from .constants import (
    MODEL_TEMPERATURE, MAX_FEEDBACK_ROUNDS,
    MAX_IDENTICAL_ACTION_REPEATS, get_platform_info,
)
from .parser import parse_actions, extract_text_outside_blocks
from .actions import ActionExecutor, agent_print
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

    def __init__(self, client, model, workdir=".", tools=True, skills=False,
                 system_prompt=None, name="uhu", confirm_callback=None,
                 auto_approve_safe=False, stream=True, temperature=None,
                 max_feedback_rounds=None, config_filename="agentic-core.json",
                 on_chunk=None):
        self.client = client
        self.model = model
        self.workdir = workdir
        self.tools = tools
        self.skills = skills
        self.stream = stream
        self.temperature = temperature if temperature is not None else MODEL_TEMPERATURE
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

                # Parse actions
                actions = parse_actions(response_text)

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

                # Store assistant response
                self.history.append({"role": "assistant", "content": response_text})

                # Execute actions
                observations = self.executor.execute_actions(actions)

                if not observations:
                    # No observations (all skipped) — nudge the model
                    observations = ["[All actions were skipped. Please try a different approach.]"]

                # Feed observations back as a user message
                obs_text = "\n\n".join(observations)
                self.history.append({"role": "user", "content": obs_text})

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
        """Call the model and return the full response text."""
        if self.stream:
            return self._call_model_stream()
        return self._call_model_nonstream()

    def _call_model_stream(self):
        """Stream the model response, calling on_chunk for each piece.

        Raises AgenticInterrupted on Ctrl+C with partial text collected so far.
        """
        chunks = []
        try:
            response = self.client.chat(
                model=self.model,
                messages=self.history,
                stream=True,
                options={"temperature": self.temperature},
            )
            for chunk in response:
                text = ""
                if isinstance(chunk, dict):
                    text = chunk.get("message", {}).get("content", "")
                elif hasattr(chunk, "message"):
                    text = chunk.message.content
                if text:
                    chunks.append(text)
                    if self.on_chunk:
                        self.on_chunk(text)
                    else:
                        sys.stdout.write(text)
                        sys.stdout.flush()
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
            chunks.append(error_msg)
        if not self.on_chunk:
            print()  # newline after streaming
        return "".join(chunks)

    def _call_model_nonstream(self):
        """Get the full model response without streaming.

        Raises AgenticInterrupted on Ctrl+C.
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=self.history,
                stream=False,
                options={"temperature": self.temperature},
            )
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return response.message.content
        except KeyboardInterrupt:
            raise AgenticInterrupted("", phase="streaming")
        except Exception as e:
            return f"[Model error: {e}]"

    @staticmethod
    def _action_signature(actions):
        """Create a hashable signature of actions for loop detection."""
        parts = []
        for a in actions:
            parts.append(f"{a['type']}:{a.get('path', '')}:{a.get('code', '')[:200]}")
        return "|".join(parts)

    def reset(self):
        """Clear conversation history, keeping only the system prompt."""
        self.history = [self.history[0]]
        self._last_action_signature = None
        self._repeat_count = 0
