"""Base classes for skills — kept separate to avoid circular imports."""

import json


class Skill:
    """Base class for all skills.

    A skill is a structured development workflow that the model can invoke.
    Skills differ from tools: tools are atomic API operations, while skills
    are multi-step workflows that gather context and guide the model.

    Attributes:
        name: Unique skill identifier (used in **SKILL:`name`** blocks).
        description: One-line human-readable description.
        system_prompt: Detailed instructions appended to the system prompt.
        parameters: Dict of parameter_name -> {"type", "required", "description"}.
    """
    name = ""
    description = ""
    system_prompt = ""
    parameters = {}
    auto_approve = True

    def execute(self, params, workdir=None, session=None):
        """Execute the skill with the given parameters.

        Returns:
            A string observation to be fed back to the model.
        """
        raise NotImplementedError


class PromptOnlySkill(Skill):
    """A skill defined purely by a system prompt — no execution logic.

    When executed, returns a confirmation that the skill was invoked.
    The model's system prompt already contains the skill's instructions,
    so it knows how to respond.
    """

    def __init__(self, name, description, system_prompt, parameters=None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.parameters = parameters or {}

    def execute(self, params, workdir=None, session=None):
        param_str = json.dumps(params, ensure_ascii=False) if params else "{}"
        return f"[Skill {self.name} invoked]\nParameters: {param_str}\n{self.system_prompt}"
