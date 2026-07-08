"""Skill registry and base classes."""

from .base import Skill, PromptOnlySkill

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
    for s in skills:
        parts.append(s.system_prompt)
        parts.append("")
    return "\n".join(parts)


__all__ = ["Skill", "PromptOnlySkill", "register", "get", "all_skills", "skills_system_prompt"]
