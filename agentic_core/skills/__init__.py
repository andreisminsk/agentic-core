"""Skill registry and base classes."""

from .base import Skill, PromptOnlySkill
from .markdown_skill import MarkdownSkill, load_skills_from_dir, parse_skill_md

_registry = {}


def register(skill_instance):
    """Register a skill instance."""
    _registry[skill_instance.name] = skill_instance


def get(name):
    """Get a skill by name."""
    return _registry.get(name)


def all_skills():
    """Return all registered skill instances."""
    return list(_registry.values())


def skills_system_prompt(enabled_names=None):
    """Build the system prompt section for enabled skills."""
    skills = all_skills() if enabled_names is None else \
        [s for s in all_skills() if s.name in enabled_names]
    if not skills:
        return ""
    parts = [
        "You have access to the following skills. To invoke a skill, use this format:",
        "",
        "**SKILL:`code_review`**",
        "```json",
        '{"path": "src/app.py"}',
        "```",
        "**EOF:`code_review`**",
        "",
        "The ```json block with parameters is REQUIRED.",
        "The EOF path must match the skill name.",
        "",
        "Available skills:",
        "",
    ]
    # Lazy loading: the prompt carries only name + description +
    # triggers. Full instructions return as the observation on
    # invocation — prompt cost stays flat as the skill library grows.
    for s in skills:
        trigger_line = ""
        if getattr(s, "triggers", None):
            trigger_line = f"\nTriggers: {', '.join(s.triggers)}"
        parts.append(f"**{s.name}** — {s.description}{trigger_line}")
        parts.append("")
    return "\n".join(parts)


__all__ = ["Skill", "PromptOnlySkill", "MarkdownSkill", "register", "get",
           "all_skills", "skills_system_prompt", "load_skills_from_dir",
           "parse_skill_md"]
