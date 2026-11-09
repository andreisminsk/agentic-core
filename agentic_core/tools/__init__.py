"""Tool registry and base class for agentic tools.

Native tools (Phase: native solicitation): tools may declare a flat
`parameters` dict — [{pname: {type, required, description, enum, items}}] —
matching the ```json params of the text protocol. tools_schemas() converts it
into the /chat `tools` wire array for native-function-calling models.
Tools without `parameters` are prose-only: they stay reachable via
**TOOL:** blocks and are skipped from the wire array (empty-schema wire 400s
are the field-proven failure mode — uhu 2026-11-08).
"""

from .fs import (
    ReadFileTool, WriteFileTool, ListFilesTool,
    FindFileTool, SearchInFilesTool, PeekFileTool,
    MkdirTool, CopyFileTool, MoveFileTool, ReplaceInFileTool,
)
from .web import (
    WebSearchTool, WebFetchTool, HttpRequestTool, TimeNowTool,
)
from .calculator import CalculatorTool
from .image_analysis import ImageAnalysisTool


class Tool:
    """Base class for all tools.

    Attributes:
        name: Unique tool identifier (used in **TOOL:`name`** blocks).
        description: One-line human-readable description.
        system_prompt: Instructions appended to the system prompt describing
            how and when to use this tool.
        auto_approve: If True, this tool is auto-approved (no confirmation).
        parameters: Optional flat params dict for native schemas —
            {name: {"type": "string|int|number|bool|object|array",
                    "required": bool, "description": str,
                    "enum": [...], "items": {...}}}.
            None/absent = prose-only tool (excluded from native schemas).
    """
    name = ""
    description = ""
    system_prompt = ""
    auto_approve = True
    parameters = None

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
        # Lazy MCP activation: mcp_* tools stay REGISTERED (execution
        # + enablement gating) but are excluded from the prompt — the
        # mcp_usage meta-tool serves their docs on demand. Hosts
        # without MCP are unaffected (no mcp_ names).
        if t.name.startswith("mcp_") and t.name != "mcp_usage":
            continue
        parts.append(t.system_prompt)
        parts.append("")
    return "\n".join(parts)


# ── Native tools (schemas for the /chat tools param) ────────────────────

def _flat_params_to_schema(name, description, params):
    """Convert the flat parameters format ({name: {type, description, required,
    enum, items}}) into a JSON-Schema function definition for the tools param.

    Mirrors uhu's converter: type aliases are guarded ("float" → "number") so
    one bad declaration can't get the whole tools array rejected with a wire 400.
    """
    params = params or {}
    properties = {}
    required = []
    for pname, pdef in params.items():
        if not isinstance(pdef, dict):
            continue
        t = pdef.get("type", "string")
        if t == "float":
            t = "number"  # JSON Schema has no float (uhu field fix, mistral-large-4)
        prop = {"type": t}
        if pdef.get("description"):
            prop["description"] = pdef["description"]
        if pdef.get("enum"):
            prop["enum"] = list(pdef["enum"])
        if pdef.get("items"):
            prop["items"] = pdef["items"]
        properties[pname] = prop
        if pdef.get("required"):
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tools_schemas(enabled_names=None):
    """Build the native tools array for the /chat tools param.

    Tools with a declared `parameters` attribute are converted via
    _flat_params_to_schema (MCPTool always qualifies — its parameters are
    built from the server-declared input_schema, so an empty dict means the
    SERVER declared the tool parameterless, not a docs bug). Tools without
    `parameters` (prose-only) are skipped: they stay reachable via
    **TOOL:** blocks in the text protocol.
    """
    tools = all_tools() if enabled_names is None else \
        [t for t in all_tools() if t.name in enabled_names]
    out = []
    for t in tools:
        # Registered mcp_* tools TRAVEL AS SCHEMAS in native modes: the
        # lazy-activation prompt skip (tools_system_prompt) serves docs on
        # demand, but native-calling models never see prompt docs — their
        # channel is the tools array. If the host registered them, wire them.
        if getattr(t, "parameters", None) is None and not hasattr(t, "input_schema"):
            continue
        try:
            out.append(_flat_params_to_schema(
                t.name, t.description, getattr(t, "parameters", None)))
        except Exception:
            continue  # prose-reachable fallback stays
    return out


def tools_native_prompt(channel="tools"):
    """Compact tools section for native-tools mode.

    Replaces the per-tool prose listing — schemas travel via the tools
    param. channel="hybrid" adds the text-block preference for file
    content (transparency play; uhu session-2: file tools stay included
    as BACKUP — exclusion cornered models into channel mixing).
    """
    parts = [
        "TOOLS: available via native tool calls — names, descriptions, and parameters "
        "are provided by the runtime. Invoke native tools ONLY via tool calls, not text. "
        "Conversely, FILE:/WRITE:/EDIT:/RUN: action blocks are NOT native tools: "
        "emit them only as normal assistant text, never inside a tool-call name or arguments.\n",
    ]
    if channel == "hybrid":
        parts.append(
            "- File content: PREFER text action blocks emitted as normal assistant "
            "text — the transparent path (diffs, review, rollback, caching): "
            "READ with FILE:, CREATE with WRITE:, MODIFY with EDIT:. Do not "
            "wrap these blocks in native tool calls. If you choose the native "
            "channel instead, call the exact backup tool names: read_file, "
            "write_file, replace_in_file.\n"
        )
    else:
        parts.append(
            "- File content: read_file to study, write_file to create, replace_in_file to modify.\n"
        )
    return "".join(parts)


# Register built-in file tools
for _t in (ReadFileTool(), WriteFileTool(), ListFilesTool(),
           FindFileTool(), SearchInFilesTool(), PeekFileTool(),
           MkdirTool(), CopyFileTool(), MoveFileTool(),
           ReplaceInFileTool(),
           WebSearchTool(), WebFetchTool(), HttpRequestTool(), TimeNowTool(),
           CalculatorTool(), ImageAnalysisTool()):
    register(_t)
