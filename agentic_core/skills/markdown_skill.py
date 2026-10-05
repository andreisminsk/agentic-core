"""MarkdownSkill — a skill defined by a SKILL.md file (SKILLS-ARCH.md §3).

Directory layout:
    <skills_root>/<category>/<skill-name>/
        SKILL.md          required — frontmatter + instructions
        scripts/         optional — executable Python scripts
        references/      optional — context documents

The system prompt carries only name/description/triggers (lazy
loading); full instructions return as the observation on invocation.
Script references in the instructions are rewritten to absolute paths
at load time so the model's RUN blocks work unchanged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys


class MarkdownSkill:
    """A skill loaded from a SKILL.md directory."""

    def __init__(self, name, description, system_prompt, parameters=None,
                 scripts=None, references=None, skill_dir=None, triggers=None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.parameters = parameters or {}
        self.scripts = scripts or {}       # script name -> abs path
        self.references = references or []  # abs paths
        self.skill_dir = skill_dir
        self.triggers = triggers or []

    def execute(self, params, workdir=None, session=None):
        """Return the skill's instructions as the observation.

        Prompt-only by design (SKILLS-ARCH.md §5): the model executes
        any scripts through its confined RUN tool; the skill itself
        never spawns a subprocess.
        """
        param_str = json.dumps(params, ensure_ascii=False) if params else "{}"
        parts = [
            f"[Skill {self.name} invoked]",
            f"Parameters: {param_str}",
            "",
            self.system_prompt,
        ]
        if self.references:
            parts.append("\nReference documents:")
            for ref in self.references:
                parts.append(f"- {ref}")
        return "\n".join(parts)


# ── SKILL.md parsing ────────────────────────────────────────────────────

def _parse_yaml_frontmatter(content: str):
    """Parse YAML frontmatter (--- delimited). Returns (dict, body)."""
    if not content.startswith("---"):
        return None, content
    lines = content.split("\n")
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None, content
    fm_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:]).strip()

    fm = {}
    current_key = None
    current_list = None
    for line in fm_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("  - ") and current_key:
            if current_list is None:
                current_list = []
            current_list.append(stripped[2:].strip().strip('"').strip("'"))
            fm[current_key] = current_list
            continue
        m = re.match(r"^(\w[\w-]*)\s*:\s*(.*)", stripped)
        if m:
            key = m.group(1).lower()
            value = m.group(2).strip()
            if value:
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                fm[key] = value
            else:
                fm[key] = None
            current_key = key
            current_list = None if value else []
    return fm, body


def _parse_markdown_sections(content: str) -> dict:
    """Split markdown into sections by ## headings (name -> content)."""
    sections = {}
    current = None
    current_lines = []
    for line in content.split("\n"):
        m = re.match(r"^##\s+(.+)", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(current_lines).strip()
            current = m.group(1).strip().lower()
            current_lines = []
        elif current is not None:
            current_lines.append(line)
    if current is not None:
        sections[current] = "\n".join(current_lines).strip()
    return sections


def _rewrite_script_refs(text: str, scripts: dict) -> str:
    """Rewrite short script references to absolute paths.

    `python scripts/fetch_news.py` → `python C:/.../scripts/fetch_news.py`
    so the model's RUN blocks work unchanged (SKILLS-ARCH.md §5).
    Longest match wins; refs already inside a resolved path are left
    alone.
    """
    for sname, spath in sorted(scripts.items(), key=lambda kv: -len(kv[1])):
        patterns = [
            f"scripts/{sname}.py",
            f"scripts\\{sname}.py",
            f"{sname}.py",
        ]
        for pattern in patterns:
            pos = text.find(pattern)
            while pos != -1:
                before = text[:pos]
                # Skip when part of a longer path (preceded by a
                # separator or word char)
                if pos > 0 and before[-1] not in (' ', '"', "'", "=", "|",
                                                  ";", "&", "`", "("):
                    break
                # Skip when already the resolved path
                if before.endswith(spath[:-len(pattern)] if spath.endswith(pattern) else "\x00"):
                    break
                text = text[:pos] + spath + text[pos + len(pattern):]
                pos = text.find(pattern, pos + len(spath))
    return text


def parse_skill_md(skill_md_path: str) -> MarkdownSkill | None:
    """Parse one SKILL.md into a MarkdownSkill. None on failure."""
    try:
        with open(skill_md_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    skill_dir = os.path.dirname(os.path.abspath(skill_md_path))
    frontmatter, body = _parse_yaml_frontmatter(content)

    name = ""
    description = ""
    instructions = ""
    sections = {}

    if frontmatter:
        name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()
        triggers = frontmatter.get("triggers")
        if not isinstance(triggers, list):
            triggers = []
        instructions = body
        sections = _parse_markdown_sections(body)
    else:
        # Fallback: markdown-section format
        name_match = re.match(r"^#\s+(?:Skill:\s*)?(.+)", content, re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
        sections = _parse_markdown_sections(content)
        description = (sections.get("description", "").split("\n")[0].strip()
                       if sections.get("description") else "")
        instructions = sections.get("instructions") or content
        triggers = []

    if not name:
        name = os.path.basename(skill_dir)
    name = name.lower().replace(" ", "-")
    if not instructions:
        return None

    # Discover scripts/ and references/
    scripts = {}
    scripts_dir = os.path.join(skill_dir, "scripts")
    if os.path.isdir(scripts_dir):
        for fn in sorted(os.listdir(scripts_dir)):
            if fn.endswith(".py") and not fn.startswith("_"):
                scripts[fn[:-3]] = os.path.join(scripts_dir, fn)

    references = []
    refs_dir = os.path.join(skill_dir, "references")
    if os.path.isdir(refs_dir):
        for fn in sorted(os.listdir(refs_dir)):
            if not fn.startswith(".") and os.path.isfile(os.path.join(refs_dir, fn)):
                references.append(os.path.join(refs_dir, fn))

    # Build the prompt: instructions + examples/scripts sections
    prompt_parts = [instructions]
    if sections.get("examples"):
        prompt_parts.append("\nExamples:\n" + sections["examples"])
    system_prompt = "\n".join(prompt_parts)

    # Rewrite script refs to absolute paths
    if scripts:
        system_prompt = _rewrite_script_refs(system_prompt, scripts)

    return MarkdownSkill(
        name=name,
        description=description,
        system_prompt=system_prompt,
        parameters={},
        scripts=scripts,
        references=references,
        skill_dir=skill_dir,
        triggers=triggers,
    )


def load_skills_from_dir(skills_dir: str) -> tuple[int, list[str]]:
    """Load custom skills from a directory tree.

    Layout: <skills_dir>/<category>/<skill-name>/SKILL.md — the
    category level is optional (a skill dir may sit at the top level).

    Returns (loaded_count, errors).
    """
    if not os.path.isdir(skills_dir):
        return 0, [f"Skills directory not found: {skills_dir}"]

    loaded = 0
    errors = []

    def _try_skill_dir(path: str) -> MarkdownSkill | None:
        skill_md = os.path.join(path, "SKILL.md")
        if not os.path.isfile(skill_md):
            return None
        skill = parse_skill_md(skill_md)
        if skill is None or not skill.name:
            errors.append(f"{skill_md}: could not parse skill")
            return None
        return skill

    def _scan(dir_path: str, depth: int = 0):
        nonlocal loaded
        if depth > 2:
            return
        for entry in sorted(os.listdir(dir_path)):
            if entry.startswith(".") or entry == "__pycache__":
                continue
            entry_path = os.path.join(dir_path, entry)
            if not os.path.isdir(entry_path):
                continue
            skill = _try_skill_dir(entry_path)
            if skill is not None:
                from . import register
                register(skill)
                loaded += 1
            else:
                _scan(entry_path, depth + 1)

    _scan(skills_dir)
    return loaded, errors
