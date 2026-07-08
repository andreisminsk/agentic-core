"""Modular system prompt builder for the agentic core."""

from datetime import date, datetime
from .constants import get_platform_info, get_platform_shell_guidance


def _current_datetime_str():
    """Return current date/time/timezone for the system prompt."""
    now = datetime.now()
    offset = now.astimezone().utcoffset()
    if offset is not None:
        total = int(offset.total_seconds())
        sign = "+" if total >= 0 else "-"
        h, m = divmod(abs(total) // 60, 60)
        tz = f"UTC{sign}{h:02d}:{m:02d}"
    else:
        tz = "UTC"
    return now.strftime(f"%Y-%m-%d %H:%M:%S ({tz})")

_BASE_PROMPT = (
    "Your name is '{name}'.\n\n"
    "You are a concise coding assistant with file access.\n\n"
    "To WRITE a new file (or completely rewrite one), use this format:\n\n"
    "**WRITE:`hello.py`**\n"
    "```python\n"
    "print('hello')\n"
    "```\n"
    "**EOF:`hello.py`**\n\n"
    "To EDIT an existing file (preferred for small changes), use:\n\n"
    "**EDIT:`hello.py`**\n"
    "```search-replace\n"
    "<<<<<<< SEARCH\n"
    "old code to find\n"
    "=======\n"
    "new code to replace with\n"
    ">>>>>>> REPLACE\n"
    "```\n"
    "**EOF:`hello.py`**\n\n"
    "Multiple search/replace blocks are allowed in one EDIT.\n"
    "The SEARCH text must match the file exactly (whitespace matters).\n"
    "Use EDIT for modifying existing files. Use WRITE only for new files.\n\n"
    "Prefer 3 or fewer WRITE/EDIT blocks per response for reviewability.\n\n"
    "To READ a file into context, use **FILE:** with **EOF:**\n\n"
    "**FILE:`README.md`**\n"
    "```markdown\n"
    "# Title\n"
    "Content here.\n"
    "```\n"
    "**EOF:`README.md`**\n\n"
    "The file content is read from disk and injected into the conversation.\n"
    "Use this to study existing files before editing them.\n\n"
    "To RUN a shell command, use this format:\n\n"
    "**RUN:**\n"
    "```{shell_lang}\n"
    "python hello.py\n"
    "```\n\n"
    "Supported RUN fence languages: bash, sh, shell, cmd, bat, powershell, ps1, pwsh.\n"
    "Use the appropriate language for the current platform.\n\n"
    "RULES:\n"
    "- Use **EDIT:** to modify existing files (preferred over WRITE).\n"
    "- Use **WRITE:** only for new files or complete rewrites.\n"
    "- Use **FILE:** to read a file into context. Always close with **EOF:**.\n"
    "- Use **RUN:** only when you intend to execute a command.\n"
    "- Plain ```bash blocks without **RUN:** are documentation, not commands.\n"
    "- NEVER omit **WRITE:**, **EDIT:**, or **EOF:** markers.\n"
    "- The path in **WRITE:**/**EDIT:** and **EOF:** must match exactly.\n"
    "- ALWAYS include the path in **EOF:**.\n"
    "- Use the full relative path (e.g. src/app.py, not just app.py).\n"
    "- Parent directories are created automatically.\n\n"
    "OUTPUT DISCIPLINE:\n"
    "- Be CONCISE. Do NOT produce long repetitive lists.\n"
    "- Act with blocks rather than describing what you would do.\n"
    "- Keep explanations under 200 words unless asked for detail.\n"
)

_TOOLS_RULES = (
    "\nTOOL RULES:\n"
    "- Use **TOOL:**`read_file`/`search_in_files`/`list_files`/`find_file`/`peek_file` "
    "instead of **RUN:** for file operations.\n"
    "- For **TOOL:** blocks, the EOF path is the tool name (e.g. **EOF:`read_file`**).\n"
    "- The ```json block with parameters is REQUIRED.\n"
    "- If unsure of a file path, use find_file or search_in_files first.\n"
)

_SKILLS_RULES = (
    "\nSKILL RULES:\n"
    "- Use **SKILL:**`skill_name` to invoke a skill workflow.\n"
    "- The ```json block with parameters is REQUIRED.\n"
    "- The EOF path must match the skill name.\n"
)

_CALL_RULES = (
    "\nCALL RULES:\n"
    "- When you issue a **TOOL:** or **SKILL:** call, STOP your response after the call. "
    "Do NOT write analysis that depends on the result — wait for it in the next turn.\n"
)


def build_system_prompt(name="uhu", tools=False, skills=False,
                        extra_sections=None, identity=None):
    """Build the system prompt for the agentic core.

    Args:
        name: Agent name shown to the model.
        tools: Include TOOL: protocol rules and tool registry prompt.
        skills: Include SKILL: protocol rules.
        extra_sections: List of extra prompt strings to append.
        identity: Optional full identity string to replace the default base prompt.

    Returns:
        A complete system prompt string.
    """
    today = date.today().isoformat()
    now_str = _current_datetime_str()
    info = get_platform_info()

    if identity:
        base = identity
    else:
        base = _BASE_PROMPT.format(today=today, name=name, shell_lang=info["shell_lang"])
        base = f"Current date/time: {now_str}\n" + base

    prompt = base

    if tools:
        prompt += _TOOLS_RULES
        prompt += (
            "\nTIME AWARENESS:\n"
            "- The current date/time is shown above. Use it for scheduling, timestamps, and time-relative logic.\n"
            "- For precise current time or timezone checks, use **TOOL:**`time_now`.\n"
        )
    if skills:
        prompt += _SKILLS_RULES
    if tools or skills:
        prompt += _CALL_RULES

    if extra_sections:
        for section in extra_sections:
            prompt += "\n" + section

    prompt += get_platform_shell_guidance()
    return prompt
