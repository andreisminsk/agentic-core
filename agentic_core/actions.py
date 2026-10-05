"""Action execution: WRITE, EDIT, FILE, RUN, TOOL, SKILL blocks from LLM output.

ActionExecutor orchestrates action dispatch and confirmation. Safety checking
is provided by CommandSafety (safety.py), file operations by FileOperations
(file_ops.py), and shared utilities by utils.py.
"""

import json
import os
import re
import subprocess
import sys
import time

from .constants import (
    MAX_OBSERVATION_CHARS, MAX_READ_OBSERVATION_CHARS,
    MAX_TOOL_OBSERVATION_CHARS, MAX_TOTAL_OBSERVATION_CHARS,
    MAX_CONSOLE_DISPLAY_CHARS, MAX_BATCH_WRITES,
)
from .utils import agent_print, tool_print, _truncate
from .safety import CommandSafety, _fix_win_backslash_quote
from .file_ops import FileOperations


class ActionExecutor(CommandSafety, FileOperations):
    """Executes parsed action blocks and returns observations for the model."""

    # Names that are obviously placeholders, not real file paths or tool/skill names
    _PLACEHOLDER_NAMES = {
        "path", "name", "filepath", "filename", "file_path", "file_name",
        "skill", "skill_name",
    }

    def __init__(self, workdir=".", confirm_callback=None, auto_approve_safe=False,
                 config_filename="agentic-core.json", run_timeout=120,
                 stop_check=None):
        self.workdir = os.path.abspath(workdir)
        self.confirm_callback = confirm_callback or self._default_confirm
        self.auto_approve_safe = auto_approve_safe
        self.config_filename = config_filename
        # RUN command timeout (seconds). Hardcoded 120 was too tight for
        # builds, test suites, and long batch jobs — hosts can raise it.
        self.run_timeout = max(int(run_timeout or 0), 1)
        # Optional callable () -> bool, polled every 0.5s during RUN
        # execution. When it returns True the process TREE is killed and
        # KeyboardInterrupt is raised (the session converts it to
        # AgenticInterrupted — the same path as Ctrl+C). Hosts wire a
        # stop event; None = today's blocking behavior.
        self.stop_check = stop_check
        # Session-level auto-approval state (reset per session)
        self.auto_all = False
        self.auto_writes = set()       # paths auto-approved this session
        self.auto_runs = set()        # command prefixes auto-approved this session
        self.always_writes = set()    # paths auto-approved persistently
        self.always_runs = set()      # command prefixes auto-approved persistently
        self._load_persistent_config()

    def _config_path(self):
        return os.path.join(self.workdir, self.config_filename)

    def _load_persistent_config(self):
        """Load persistent always-writes/always-runs from config file."""
        path = self._config_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.always_writes.update(cfg.get("always_writes", []))
            self.always_runs.update(cfg.get("always_runs", []))
        except Exception:
            pass

    def _save_persistent_config(self):
        """Save always-writes/always-runs to config file."""
        path = self._config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "always_writes": sorted(self.always_writes),
                    "always_runs": sorted(self.always_runs),
                }, f, indent=2)
        except Exception:
            pass

    def _default_confirm(self, atype, details):
        """Interactive confirmation with y/N/auto/all/always/d options.

        Options:
          y/yes    — approve once
          n/no     — reject (default)
          auto     — approve this path/command for the rest of this session
          all      — approve ALL actions for the rest of this session
          always   — approve this path/command persistently (saved to config)
          d/diff   — show diff/details before deciding
        """
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return True  # non-interactive: auto-approve

        # Build prompt label
        if atype == "run":
            label = f"[RUN] {details.get('command', '')[:100]}"
            key = details.get("command", "").strip()
            path = None
        elif atype in ("write", "edit"):
            label = f"[{atype.upper()}] {details.get('path', '')}"
            path = details.get("path", "")
            key = path
        elif atype in ("tool", "skill"):
            label = f"[{atype.upper()}] {details.get('name', '')}"
            path = None
            key = None
        else:
            label = f"[{atype.upper()}] {details}"
            key = None
            path = None

        diff_text = details.get("diff")

        while True:
            # Check auto-approval state
            if self.auto_all:
                agent_print(f"[auto-all] {label}")
                return True
            if path and path in self.auto_writes:
                agent_print(f"[auto-write: {path}] {label}")
                return True
            if path and path in self.always_writes:
                agent_print(f"[always-write: {path}] {label}")
                return True
            if key and atype == "run":
                for prefix in self.auto_runs | self.always_runs:
                    if key == prefix or key.startswith(prefix + " "):
                        source = "always" if prefix in self.always_runs else "auto"
                        agent_print(f"[{source}-run: {prefix}] {label}")
                        return True

            try:
                ans = input(f"  {label}\n  Allow? [y/N/auto/all/always/d]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False

            if ans in ("y", "yes"):
                return True
            elif ans in ("n", "no", ""):
                return False
            elif ans == "auto":
                if path:
                    self.auto_writes.add(path)
                    agent_print(f"[Auto-write: {path} — this session]")
                elif key and atype == "run":
                    self.auto_runs.add(key)
                    agent_print(f"[Auto-run: {key} — this session]")
                return True
            elif ans == "all":
                self.auto_all = True
                agent_print("[Auto-all enabled — all actions auto-approved this session]")
                return True
            elif ans == "always":
                if path:
                    self.always_writes.add(path)
                    self.auto_writes.add(path)
                    self._save_persistent_config()
                    agent_print(f"[Always-write: {path} — saved for future sessions]")
                elif key and atype == "run":
                    self.always_runs.add(key)
                    self.auto_runs.add(key)
                    self._save_persistent_config()
                    agent_print(f"[Always-run: {key} — saved for future sessions]")
                return True
            elif ans in ("d", "diff", "details"):
                if diff_text:
                    self._show_diff(diff_text)
                elif atype in ("tool", "skill"):
                    import json as _json
                    params = details.get("params", {})
                    agent_print(f"  Parameters: {_json.dumps(params, indent=2, ensure_ascii=False)}")
                elif atype == "run":
                    agent_print(f"  Full command: {details.get('command', '')}")
                    agent_print(f"  Safety level: {details.get('level', 'unknown')}")
                elif atype in ("write", "edit"):
                    agent_print(f"  File: {details.get('path', '')}")
                else:
                    agent_print("[No details available]")
                continue
            return False

    @staticmethod
    def _show_diff(diff_text):
        """Display a diff with basic color coding."""
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            except Exception:
                pass
        for line in diff_text.splitlines():
            if sys.stdout.isatty():
                if line.startswith('+') and not line.startswith('+++'):
                    print(f"\033[32m{line}\033[0m")
                elif line.startswith('-') and not line.startswith('---'):
                    print(f"\033[31m{line}\033[0m")
                elif line.startswith('@@'):
                    print(f"\033[36m{line}\033[0m")
                else:
                    print(line)
            else:
                print(line)

    def execute_actions(self, actions):
        """Execute a list of parsed actions. Returns list of observation strings.

        Raises AgenticInterrupted if the user presses Ctrl+C during execution.
        Observations collected before the interrupt are preserved in the
        exception's partial_text.
        """
        from .exceptions import AgenticInterrupted

        observations = []
        total_chars = 0
        write_count = 0
        modified = set()   # paths modified by write/edit
        created = set()     # paths created by write

        for action in actions:
            atype = action["type"]
            action_path = action.get("path") or action.get("name", "")

            # Skip placeholder paths (model used a template name instead of real path)
            if action_path and action_path.lower().strip() in self._PLACEHOLDER_NAMES:
                observations.append(
                    f"[SYSTEM WARNING: {atype.upper()} block uses placeholder "
                    f"'{action_path}' instead of a real file path or tool name. "
                    f"Use actual values like **{atype.upper()}:`src/app.py`**]"
                )
                continue

            # Batch write limit
            if atype in ("write", "edit"):
                write_count += 1
                if write_count > MAX_BATCH_WRITES:
                    observations.append(
                        f"[Batch limit reached: {MAX_BATCH_WRITES} writes. "
                        f"Continue in next response to apply remaining changes.]"
                    )
                    break

                # Track pre-edit state for rollback
                path = action.get("path", "")
                if path and path not in modified and path not in created:
                    full_path = os.path.join(self.workdir, path)
                    if os.path.isfile(full_path):
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                                self._save_pre_edit(path, f.read())
                            modified.add(path)
                        except Exception:
                            modified.add(path)
                    else:
                        created.add(path)

            # Confirmation
            try:
                if not self._should_approve(atype, action):
                    observations.append(f"[{atype} skipped: not approved by user]")
                    continue
            except KeyboardInterrupt:
                raise AgenticInterrupted(
                    "\n\n".join(observations) if observations else "",
                    phase="confirmation",
                )

            try:
                obs = self._execute_one(action)
            except KeyboardInterrupt:
                observations.append(f"[{atype} interrupted by user]")
                raise AgenticInterrupted(
                    "\n\n".join(observations),
                    phase="actions",
                )

            if obs:
                # Truncate based on type
                if atype == "file":
                    obs = _truncate(obs, MAX_READ_OBSERVATION_CHARS)
                elif atype in ("tool", "skill"):
                    obs = _truncate(obs, MAX_TOOL_OBSERVATION_CHARS)
                else:
                    obs = _truncate(obs, MAX_OBSERVATION_CHARS)

                total_chars += len(obs)
                if total_chars > MAX_TOTAL_OBSERVATION_CHARS:
                    observations.append("[Total observation limit reached — stopping.]")
                    break
                observations.append(obs)

        return observations

    def _should_approve(self, atype, action):
        """Check if an action should be approved.

        Policy:
          - file:        always approved (read-only)
          - run:         blocked → reject; safe + auto_approve_safe → approve;
                        otherwise → confirm_callback
          - write/edit:  if auto_approve_safe → approve; otherwise → confirm_callback
          - tool/skill:  if the tool/skill has auto_approve=True → approve;
                        otherwise → confirm_callback
        """
        if atype == "file":
            return True  # read-only, always safe

        if atype == "run":
            cmd = action.get("code", "")
            level, msg = self._check_command_safety(cmd)
            if level == "blocked":
                agent_print(msg)
                return False
            if level == "safe" and self.auto_approve_safe:
                return True
            return self.confirm_callback(atype, {"command": cmd, "level": level, "message": msg})

        if atype in ("write", "edit"):
            if self.auto_approve_safe:
                return True
            details = {"path": action.get("path", "")}
            if atype == "edit":
                details["diff"] = self._preview_edit_diff(action)
            return self.confirm_callback(atype, details)

        if atype in ("tool", "skill"):
            name = action.get("path", "")
            # Check the tool/skill's own auto_approve flag
            obj = None
            if atype == "tool":
                from .tools import get as get_tool
                obj = get_tool(name)
            else:
                from .skills import get as get_skill
                obj = get_skill(name)
            if obj is not None and getattr(obj, "auto_approve", True):
                return True
            return self.confirm_callback(atype, {"name": name, "params": action.get("params", {})})

        return True

    def _execute_one(self, action):
        """Execute a single action and return its observation string."""
        atype = action["type"]
        if atype == "write":
            return self._do_write(action)
        elif atype == "edit":
            return self._do_edit(action)
        elif atype == "file":
            return self._do_file(action)
        elif atype == "run":
            return self._do_run(action)
        elif atype == "tool":
            return self._do_tool(action)
        elif atype == "skill":
            return self._do_skill(action)
        return f"[Unknown action type: {atype}]"

    @staticmethod
    def _kill_tree(proc) -> None:
        """Kill a process and its children.

        shell=True spawns cmd.exe/bash → children; killing only the
        shell orphans them. Windows: taskkill /F /T; POSIX: killpg on
        the child's own session group (start_new_session=True).
        """
        import signal
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=10,
                )
            else:
                import os
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _drain_after_kill(proc, timeout=2.0):
        """Bounded post-kill cleanup. Two verified Windows traps when an
        orphaned grandchild still holds the pipe write-handles (e.g.
        `start "" cmd` opens a terminal that inherits them):
        1. communicate(timeout) re-enters _communicate and JOINS blocked
           reader threads — the timeout is NOT honored on that path
           (observed: 28s hang with timeout=2).
        2. stream.close() blocks on the buffer lock held by a stuck
           reader thread.
        So: touch neither readers nor streams. wait() on the process
        only (the shell is dead after the tree kill), abandon the pipes
        — reader threads are daemons and die with the orphan's EOF.
        Returns (stdout, stderr) — possibly empty."""
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
        return b"", b""

    def _do_run(self, action):
        cmd = action["code"]
        lang = action.get("lang", "bash")

        # Route PowerShell fence blocks through powershell.exe instead of cmd.exe
        if lang in ("powershell", "ps1", "pwsh"):
            if sys.platform == "win32":
                cmd = f'powershell -NoProfile -Command "{cmd}"'
            else:
                return "[PowerShell fence block used on non-Windows platform — skipping.]"

        # On Windows cmd.exe, fix odd backslashes before closing quotes
        if sys.platform == "win32":
            cmd = re.sub(r'(\\+)"', _fix_win_backslash_quote, cmd)

        # Determine fallback encoding for Windows subprocess output
        _fallback_encoding = None
        if sys.platform == "win32":
            try:
                import ctypes as _ctypes
                _oem_cp = _ctypes.windll.kernel32.GetOEMCP()
                _fallback_encoding = f'cp{_oem_cp}'
            except Exception:
                pass

        agent_print(f"[RUN: {cmd[:100]}]")
        try:
            # Stop-aware execution: Popen + poll loop instead of a single
            # blocking subprocess.run. stop_check (optional host-wired
            # callable) is polled every 0.5s; on stop the process tree is
            # killed and KeyboardInterrupt is raised — the session converts
            # it to AgenticInterrupted, exactly like Ctrl+C. Without
            # stop_check, behavior is unchanged (blocking until exit or
            # run_timeout).
            popen_kwargs = dict(
                shell=True, cwd=self.workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                executable=None if sys.platform == "win32" else "/bin/bash",
            )
            if sys.platform != "win32":
                # Own process group — killpg can never touch the host's.
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, **popen_kwargs)
            deadline = time.monotonic() + self.run_timeout
            timed_out = False
            while True:
                try:
                    out_bytes, err_bytes = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if self.stop_check is not None and self.stop_check():
                        self._kill_tree(proc)
                        self._drain_after_kill(proc)
                        raise KeyboardInterrupt()
                    if time.monotonic() > deadline:
                        timed_out = True
                        self._kill_tree(proc)
                        out_bytes, err_bytes = self._drain_after_kill(proc)
                        break
            result = subprocess.CompletedProcess(
                args=cmd, returncode=proc.returncode,
                stdout=out_bytes, stderr=err_bytes,
            )
            if timed_out:
                return (f"Error: command timed out after "
                        f"{self.run_timeout}s: {cmd[:80]}")
            # Decode output with fallback for OEM codepages
            try:
                output = result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                if _fallback_encoding:
                    try:
                        output = result.stdout.decode(_fallback_encoding, errors='replace')
                    except Exception:
                        output = result.stdout.decode('utf-8', errors='replace')
                else:
                    output = result.stdout.decode('utf-8', errors='replace')
            try:
                stderr = result.stderr.decode('utf-8')
            except UnicodeDecodeError:
                if _fallback_encoding:
                    try:
                        stderr = result.stderr.decode(_fallback_encoding, errors='replace')
                    except Exception:
                        stderr = result.stderr.decode('utf-8', errors='replace')
                else:
                    stderr = result.stderr.decode('utf-8', errors='replace')
            if stderr:
                output += stderr
            output = output.strip()
            if not output:
                output = f"[Command completed with exit code {result.returncode}]"
            else:
                output += f"\n[exit code: {result.returncode}]"
            tool_print(output[:MAX_CONSOLE_DISPLAY_CHARS])
            return output
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return f"Error running command: {e}"

    def _do_tool(self, action):
        from .tools import get as get_tool
        name = action["path"]
        params = action.get("params", {})
        json_err = action.get("json_error")
        if json_err:
            return f"Error parsing tool params for {name}: {json_err}"
        tool = get_tool(name)
        if tool is None:
            return f"Error: unknown tool: {name}"
        try:
            result = tool.execute(params, workdir=self.workdir)
            tool_print(f"[TOOL {name}]: {str(result)[:MAX_CONSOLE_DISPLAY_CHARS]}")
            return f"[TOOL {name}]\n{result}"
        except Exception as e:
            return f"Error executing tool {name}: {e}"

    def _do_skill(self, action):
        from .skills import get as get_skill
        name = action["path"]
        params = action.get("params", {})
        json_err = action.get("json_error")
        if json_err:
            return f"Error parsing skill params for {name}: {json_err}"
        skill = get_skill(name)
        if skill is None:
            hint = ""
            from .tools import get as get_tool
            if get_tool(name) is not None:
                hint = (f" ({name} is a TOOL — invoke it with a TOOL: block, "
                        f"not SKILL:)")
            return f"Error: unknown skill: {name}{hint}"
        try:
            result = skill.execute(params, workdir=self.workdir)
            tool_print(f"[SKILL {name}]: {str(result)[:MAX_CONSOLE_DISPLAY_CHARS]}")
            return f"[SKILL {name}]\n{result}"
        except Exception as e:
            return f"Error executing skill {name}: {e}"
