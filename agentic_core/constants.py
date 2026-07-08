"""Shared constants for the agentic core."""

import sys

# ── Model defaults ──────────────────────────────────────────────────────
MODEL_TEMPERATURE = 0.0

# ── Context conservation limits ────────────────────────────────────────
MAX_OBSERVATION_CHARS = 4000
MAX_READ_OBSERVATION_CHARS = 70000
MAX_TOOL_OBSERVATION_CHARS = 8000
MAX_TOTAL_OBSERVATION_CHARS = 120000
MAX_CONSOLE_DISPLAY_CHARS = 2000
MAX_BATCH_WRITES = 3

# ── Loop detection ─────────────────────────────────────────────────────
MAX_IDENTICAL_ACTION_REPEATS = 3
MAX_FEEDBACK_ROUNDS = 3

# ── Safe shell commands (auto-approved, read-only) ─────────────────────
SAFE_SHELL_COMMANDS_UNIX = {
    'ls', 'tree', 'pwd', 'cd', 'cat', 'head', 'tail', 'less', 'more',
    'file', 'stat', 'wc', 'diff', 'cmp', 'find', 'grep', 'rg', 'ag', 'ack',
    'which', 'whoami', 'hostname', 'uname', 'env', 'printenv', 'sort', 'uniq',
}
SAFE_SHELL_COMMANDS_WINDOWS = {
    'dir', 'tree', 'cd', 'type', 'more', 'fc', 'comp', 'findstr', 'rg', 'ag',
    'where', 'whoami', 'hostname', 'ver', 'set', 'print',
}
SAFE_SHELL_COMMANDS = SAFE_SHELL_COMMANDS_WINDOWS if sys.platform == 'win32' else SAFE_SHELL_COMMANDS_UNIX

# ── Blocked commands (never executed) ──────────────────────────────────
BLOCKED_COMMANDS_UNIX = {
    'shutdown', 'reboot', 'poweroff', 'halt', 'mkfs', 'dd if=',
    'rm -rf /', 'rm -rf /*',
}
BLOCKED_COMMANDS_WINDOWS = {
    'shutdown', 'format', 'del /s /q c:', 'del /s /q C:',
    'rmdir /s /q c:', 'rmdir /s /q C:',
}
BLOCKED_COMMANDS = BLOCKED_COMMANDS_WINDOWS if sys.platform == 'win32' else BLOCKED_COMMANDS_UNIX

# ── Warning commands (require confirmation even with auto-approve) ─────
WARNING_COMMANDS_UNIX = {
    'rm', 'rmdir', 'chmod', 'chown', 'kill', 'killall',
    'apt', 'apt-get', 'yum', 'dnf', 'brew', 'pip uninstall',
    'npm uninstall', 'git push', 'git reset --hard', 'git clean',
    'systemctl', 'service',
}
WARNING_COMMANDS_WINDOWS = {
    'del', 'rmdir', 'taskkill', 'sc', 'net', 'netsh',
    'pip uninstall', 'npm uninstall',
    'git push', 'git reset --hard', 'git clean',
}
WARNING_COMMANDS = WARNING_COMMANDS_WINDOWS if sys.platform == 'win32' else WARNING_COMMANDS_UNIX

# ── File extensions to skip in search/list ─────────────────────────────
SKIP_EXT = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.ogg', '.flac',
    '.exe', '.dll', '.so', '.dylib', '.o', '.obj', '.pyc', '.pyo', '.class',
    '.woff', '.woff2', '.ttf', '.eot',
    '.db', '.sqlite', '.sqlite3',
}

# ── ANSI colors ────────────────────────────────────────────────────────
ANSI_RESET = "\033[0m"
ANSI_AGENT = "\033[93m"
ANSI_TOOL = "\033[97;2;3m"


def get_platform_info():
    """Return platform-specific shell info."""
    if sys.platform == 'win32':
        return {'shell_lang': 'cmd', 'platform_label': 'win32', 'shell_label': 'cmd/powershell'}
    return {'shell_lang': 'bash', 'platform_label': sys.platform, 'shell_label': 'bash/sh'}


def get_platform_shell_guidance():
    """Return platform-specific shell guidance for the system prompt."""
    from datetime import date
    info = get_platform_info()
    today = date.today().isoformat()
    header = f"Platform: {info['platform_label']} | Shell: {info['shell_label']} | Current date: {today}\n\n"
    if sys.platform == 'win32':
        return "\n\n" + header + (
            "SHELL AND COMMAND GUIDELINES FOR WINDOWS:\n"
            "- Use Windows-native commands in RUN blocks.\n"
            "- Use `dir` instead of `ls`, `type` instead of `cat`, `findstr` instead of `grep`.\n"
            "- Use `where` instead of `which`, `set` instead of `env`, `ver` instead of `uname`.\n"
            "- Use `cmd` or `powershell` as the RUN fence language.\n"
        )
    return "\n\n" + header + (
        "SHELL AND COMMAND GUIDELINES FOR UNIX:\n"
        "- Use standard Unix commands in RUN blocks.\n"
        "- Use `bash` or `sh` as the RUN fence language.\n"
    )
