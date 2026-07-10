"""Command safety checking for shell execution in action blocks.

Extracted from actions.py — handles blocked/warning/safe classification,
shell operator detection, and base command extraction.
"""

import os
import re
import sys

from .constants import SAFE_SHELL_COMMANDS, BLOCKED_COMMANDS, WARNING_COMMANDS


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


class CommandSafety:
    """Mixin providing command safety checking for ActionExecutor.

    Class attributes:
        _SHELL_OPERATORS: Regex detecting shell chaining operators.
        _BLOCKED_PATTERNS: Regexes for commands that are ALWAYS blocked.
        _WARNING_BASE_COMMANDS: Base commands requiring explicit confirmation.
        _WARNING_SUBSTRINGS: Substrings indicating destructive operations.
    """

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
