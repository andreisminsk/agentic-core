"""Action execution: WRITE, EDIT, FILE, RUN, TOOL, SKILL blocks from LLM output."""

import json
import os
import re
import shutil
import subprocess
import sys

from .constants import (
    SAFE_SHELL_COMMANDS, BLOCKED_COMMANDS, WARNING_COMMANDS,
    MAX_OBSERVATION_CHARS, MAX_READ_OBSERVATION_CHARS,
    MAX_TOOL_OBSERVATION_CHARS, MAX_TOTAL_OBSERVATION_CHARS,
    MAX_CONSOLE_DISPLAY_CHARS, MAX_BATCH_WRITES,
    ANSI_AGENT, ANSI_RESET, ANSI_TOOL,
)
from .edit_utils import make_edit_summary, make_unified_diff
from .parser import parse_edit_content
from .matching import find_match_in_content


def _fix_win_backslash_quote(m):
    """Fix odd number of backslashes before a closing quote on Windows.

    In cmd.exe, \" is an escaped quote, so "C:\\path\\" is parsed as C:\\path"
    (invalid path). Remove one backslash from odd-length runs so the quote
    properly terminates the argument.
    """
    backslashes = m.group(1)
    if len(backslashes) % 2 == 1:
        return backslashes[:-1] + '"'
    return m.group(0)

def agent_print(*args, **kwargs):
    """Print with agent color when stdout is a TTY."""
    if sys.stdout.isatty():
        print(ANSI_AGENT, end="", flush=True)
        print(*args, **kwargs)
        print(ANSI_RESET, end="", flush=True)
    else:
        print(*args, **kwargs)


def tool_print(*args, **kwargs):
    """Print with tool color when stdout is a TTY."""
    if sys.stdout.isatty():
        print(ANSI_TOOL, end="", flush=True)
        print(*args, **kwargs)
        print(ANSI_RESET, end="", flush=True)
    else:
        print(*args, **kwargs)


def _truncate(text, max_chars):
    """Truncate text to max_chars, adding a notice if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[... truncated, {len(text)} chars total]"


class ActionExecutor:
    """Executes parsed action blocks and returns observations for the model."""

    # Names that are obviously placeholders, not real file paths or tool/skill names
    _PLACEHOLDER_NAMES = {
        "path", "name", "filepath", "filename", "file_path", "file_name",
        "skill", "skill_name",
    }

    # Shell operators that allow chaining multiple commands.
    # Includes command substitution ($(...), backticks), process substitution
    # (<(...), >(...)), and newlines — all can hide destructive commands
    # inside an otherwise "safe" base command (e.g., echo $(rm -rf /)).
    _SHELL_OPERATORS = re.compile(r'&&|\|\||[|;&]|\$\(|`|\n|\r|<\(|>\(')

    # Patterns that are ALWAYS blocked — never executed.
    _BLOCKED_PATTERNS = [
        re.compile(r'\brm\s+-[rR].*\s+/\s*$', re.IGNORECASE),      # rm -rf /
        re.compile(r'\brm\s+-[rR].*\s+/\*', re.IGNORECASE),        # rm -rf /*
        re.compile(r'\brm\s+-[rRf]*\s+/', re.IGNORECASE),          # rm -rf / anywhere (incl. inside $())
        re.compile(r'\brm\s+--recursive.*\s+/', re.IGNORECASE),    # rm --recursive /
        re.compile(r'\bdd\s+if=', re.IGNORECASE),                  # dd if=...
        re.compile(r'\bmkfs\b', re.IGNORECASE),                    # mkfs
        re.compile(r'\bshutdown\b', re.IGNORECASE),                # shutdown
        re.compile(r'\breboot\b', re.IGNORECASE),                  # reboot
        re.compile(r'\bpoweroff\b', re.IGNORECASE),                # poweroff
        re.compile(r'\bhalt\b', re.IGNORECASE),                    # halt
        re.compile(r'\bformat\s+[A-Za-z]:', re.IGNORECASE),        # format C:
        re.compile(r'\bdel\s+/s\s+/q\s+[cC]:', re.IGNORECASE),    # del /s /q C:
        re.compile(r'\brmdir\s+/s\s+/q\s+[cC]:', re.IGNORECASE),  # rmdir /s /q C:
    ]

    # Base commands that always require explicit confirmation (platform-specific)
    _WARNING_BASE_COMMANDS_UNIX = {
        'rm', 'rmdir', 'chmod', 'chown', 'kill', 'killall',
        'apt', 'apt-get', 'yum', 'dnf', 'brew',
        'systemctl', 'service',
    }
    _WARNING_BASE_COMMANDS_WINDOWS = {
        'del', 'rmdir', 'taskkill', 'sc', 'net', 'netsh',
    }
    _WARNING_BASE_COMMANDS = (
        _WARNING_BASE_COMMANDS_WINDOWS if sys.platform == 'win32'
        else _WARNING_BASE_COMMANDS_UNIX
    )
    # Substrings that indicate destructive package/git operations
    _WARNING_SUBSTRINGS = {
        'pip uninstall', 'npm uninstall',
        'git push', 'git reset --hard', 'git clean',
    }
    def __init__(self, workdir=".", confirm_callback=None, auto_approve_safe=False,
                 config_filename="agentic-core.json"):
        self.workdir = os.path.abspath(workdir)
        self.confirm_callback = confirm_callback or self._default_confirm
        self.auto_approve_safe = auto_approve_safe
        self.config_filename = config_filename
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
        import json
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
        import json
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

    def _check_command_safety(self, cmd):
        """Check a command for safety. Returns (level, message).

        level: 'blocked' — never execute
               'warning' — requires explicit confirmation even with auto-approve
               'chain'   — shell chaining detected, warn but allow with confirmation
               'safe'    — no issues
        """
        for pat in self._BLOCKED_PATTERNS:
            if pat.search(cmd):
                return ('blocked', f"Command blocked for safety: {cmd[:80]}")
        base = self._get_base_command(cmd)
        if base in BLOCKED_COMMANDS:
            return ('blocked', f"Command blocked for safety: {cmd[:80]}")
        if base in WARNING_COMMANDS or base in self._WARNING_BASE_COMMANDS:
            return ('warning', f"Destructive command requires confirmation: {cmd[:80]}")
        for substr in self._WARNING_SUBSTRINGS:
            if substr in cmd.lower():
                return ('warning', f"Destructive command requires confirmation: {cmd[:80]}")
        if self._SHELL_OPERATORS.search(cmd):
            return ('chain', f"Shell chaining detected: {cmd[:80]}")
        return ('safe', '')

    @staticmethod
    def _get_base_command(cmd):
        """Extract the base command name from a shell command string."""
        cmd = cmd.strip()
        if not cmd:
            return ""
        # Handle quoted executables: "C:\path\app.exe" args
        if cmd[0] == '"':
            end = cmd.find('"', 1)
            if end > 0:
                base = cmd[1:end]
            else:
                base = cmd[1:].split(None, 1)[0] if len(cmd) > 1 else ""
        else:
            base = cmd.split(None, 1)[0]
        # Get just the executable name without path
        base = os.path.basename(base).lower()
        # Remove common executable extensions
        for ext in ('.exe', '.cmd', '.bat', '.com'):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        return base

    def _is_safe_command(self, cmd):
        """Check if a command is considered safe/read-only (no confirmation needed).

        Commands containing shell operators (&&, ||, |, &, ;, $(), etc.) are
        never auto-approved, since a safe prefix like 'dir' could chain into a
        dangerous command like 'del /s *'.
        """
        if self._SHELL_OPERATORS.search(cmd):
            return False
        return self._get_base_command(cmd) in SAFE_SHELL_COMMANDS

    def _save_pre_edit(self, path, content):
        """Save file content before modification for potential rollback."""
        if not hasattr(self, '_pre_edit_snapshots'):
            self._pre_edit_snapshots = {}
        if path not in self._pre_edit_snapshots:
            self._pre_edit_snapshots[path] = content

    def _rollback_edits(self, modified, created):
        """Restore files to their pre-edit state and remove newly created files."""
        for path in modified:
            if path in self._pre_edit_snapshots:
                full_path = os.path.join(self.workdir, path)
                os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(self._pre_edit_snapshots[path])
        for path in created:
            full_path = os.path.join(self.workdir, path)
            if os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass

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

    def _preview_edit_diff(self, action):
        """Generate a diff preview for an EDIT action before execution."""
        path = action["path"]
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return None
        try:
            with open(full, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception:
            return None
        blocks = parse_edit_content(action["code"])
        if not blocks:
            return None
        content = original
        for search_text, replace_text in blocks:
            if not search_text:
                content = content + replace_text
                continue
            match_info = find_match_in_content(content, search_text)
            if match_info is None:
                continue
            start, end, _ = match_info
            content = content[:start] + replace_text + content[end:]
        return make_unified_diff(path, original, content)

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

    def _do_write(self, action):
        path = action["path"]
        code = action["code"]
        full = os.path.join(self.workdir, path)
        # Cache previous version
        if os.path.isfile(full):
            cache_dir = os.path.join(self.workdir, ".uhu", ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            base, ext = os.path.splitext(path)
            for i in range(1, 999):
                cache_name = f"{base}.{i}{ext}"
                cache_path = os.path.join(cache_dir, cache_name)
                if not os.path.exists(cache_path):
                    try:
                        shutil.copy2(full, cache_path)
                    except Exception:
                        pass
                    break
        # Create parent dirs
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(full, "w", encoding="utf-8") as f:
                f.write(code)
            lines = code.count('\n') + (1 if code and not code.endswith('\n') else 0)
            agent_print(f"[Wrote: {path} ({len(code)} bytes, {lines} lines)]")
            return f"[Wrote: {path} ({len(code)} bytes, {lines} lines)]"
        except Exception as e:
            return f"Error writing {path}: {e}"
    def _do_edit(self, action):
        path = action["path"]
        code = action["code"]
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return f"Error: file not found for edit: {path}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

        blocks = parse_edit_content(code)
        if not blocks:
            return f"Error: no search/replace blocks found in EDIT for {path}"

        content = original
        edits_applied = []
        failures = []
        for search_text, replace_text in blocks:
            if not search_text:
                # Insert at end
                content = content + replace_text
                edits_applied.append((len(content.split('\n')) - 1, len(content.split('\n')), 'insert', search_text, replace_text))
                continue
            match_info = find_match_in_content(content, search_text)
            if match_info is None:
                failures.append(search_text)
                continue
            start, end, quality = match_info
            content = content[:start] + replace_text + content[end:]
            edits_applied.append((start, end, quality, search_text, replace_text))

        if not edits_applied and failures:
            # All search blocks failed — show file content snippet to help model
            snippet_lines = original.split('\n')[:20]
            snippet = '\n'.join(snippet_lines)
            if len(original.split('\n')) > 20:
                snippet += f"\n... ({len(original.split(chr(10)))} lines total)"
            return (
                f"[EDIT FAILED: {path} — search text not found]\n"
                f"File content:\n{snippet}"
            )

        if failures:
            # Some blocks failed — apply what we can, warn about the rest
            failed_previews = [f[:80].replace('\n', '\\n') for f in failures]

        try:
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Error writing {path}: {e}"

        summary = make_edit_summary(path, edits_applied)
        diff = make_unified_diff(path, original, content)
        agent_print(summary)
        if failures:
            return (
                f"{summary}\n{diff}\n"
                f"[WARNING: {len(failures)} search block(s) not found: "
                f"{'; '.join(failed_previews)}]"
            )
        return f"{summary}\n{diff}"
    def _do_file(self, action):
        path = action["path"]
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            header = f"[File: {path} | {lines_count} lines]"
            tool_print(header)
            tool_print(content[:MAX_CONSOLE_DISPLAY_CHARS])
            return f"{header}\n{content}"
        except Exception as e:
            return f"Error reading {path}: {e}"
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
            # Use binary mode and decode manually to avoid UnicodeDecodeError
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                timeout=120, cwd=self.workdir,
                executable=None if sys.platform == "win32" else "/bin/bash",
            )
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
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after 120s: {cmd[:80]}"
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
            return f"Error: unknown skill: {name}"
        try:
            result = skill.execute(params, workdir=self.workdir)
            tool_print(f"[SKILL {name}]: {str(result)[:MAX_CONSOLE_DISPLAY_CHARS]}")
            return f"[SKILL {name}]\n{result}"
        except Exception as e:
            return f"Error executing skill {name}: {e}"
