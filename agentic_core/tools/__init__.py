"""Tool registry and base class for agentic tools."""

from .fs import (
    ReadFileTool, WriteFileTool, ListFilesTool,
    FindFileTool, SearchInFilesTool, PeekFileTool,
    MkdirTool, CopyFileTool, MoveFileTool, ReplaceInFileTool,
)
from .web import (
    WebSearchTool, WebFetchTool, HttpRequestTool, TimeNowTool,
)
from .calculator import CalculatorTool


class Tool:
    """Base class for all tools.

    Attributes:
        name: Unique tool identifier (used in **TOOL:`name`** blocks).
        description: One-line human-readable description.
        system_prompt: Instructions appended to the system prompt describing
            how and when to use this tool.
        auto_approve: If True, this tool is auto-approved (no confirmation).
    """
    name = ""
    description = ""
    system_prompt = ""
    auto_approve = True

    def execute(self, params, workdir=None):
        """Execute the tool with the given parameters.

        Args:
            params: Dict of parameters from the TOOL block JSON.
            workdir: Working directory for file operations.

        Returns:
            A string observation to be fed back to the model.
        """
        raise NotImplementedError


_registry = {}


def register(tool_instance):
    """Register a tool instance."""
    _registry[tool_instance.name] = tool_instance


def get(name):
    """Get a tool by name."""
    return _registry.get(name)


def all_tools():
    """Return all registered tool instances."""
    return list(_registry.values())


def tools_system_prompt(enabled_names=None):
    """Build the system prompt section for enabled tools."""
    tools = all_tools() if enabled_names is None else \
        [t for t in all_tools() if t.name in enabled_names]
    if not tools:
        return ""
    parts = [
        "You have access to the following tools. To invoke a tool, use this format:",
        "",
        "**TOOL:`read_file`**",
        "```json",
        '{"path": "src/app.py"}',
        "```",
        "**EOF:`read_file`**",
        "",
        "CRITICAL: The ```json block with parameters is REQUIRED.",
        "The EOF path must match the tool name, NOT the file path being operated on.",
        "When asked to perform multiple operations, produce ALL tool calls in a single response.",
        "Always use the full relative path for file operations.",
        "",
        "Available tools:",
        "",
    ]
    for t in tools:
        parts.append(t.system_prompt)
        parts.append("")
    return "\n".join(parts)


# Register built-in file tools
for _t in (ReadFileTool(), WriteFileTool(), ListFilesTool(),
           FindFileTool(), SearchInFilesTool(), PeekFileTool(),
           MkdirTool(), CopyFileTool(), MoveFileTool(),
           ReplaceInFileTool(),
           WebSearchTool(), WebFetchTool(), HttpRequestTool(), TimeNowTool(),
           CalculatorTool()):
    register(_t)
