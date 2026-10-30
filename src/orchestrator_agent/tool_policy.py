"""Read-only Core discovery, with capability and named-tool opt-ins."""

from collections.abc import Collection
from typing import Any

from pydantic_ai.tools import RunContext, ToolDefinition

from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL, WRITE_TOOL_NAMES

LOCAL_TOOL_NAMES = frozenset({START_WORKFLOW_FORM_TOOL, "load_capability"})


def is_read_only_core_tool(tool: ToolDefinition) -> bool:
    """Require an explicit boolean read-only declaration, keeping writes and local names out.

    Core authors are responsible for accurate annotations. The declaration controls admission;
    it does not inspect the endpoint implementation or prove that it cannot change business state.
    """
    if tool.kind == "external" or tool.name in WRITE_TOOL_NAMES or tool.name in LOCAL_TOOL_NAMES:
        return False
    annotations = (tool.metadata or {}).get("annotations")
    return isinstance(annotations, dict) and annotations.get("readOnlyHint") is True


def is_bypassed_tool(tool: ToolDefinition, bypass: Collection[str]) -> bool:
    """Allow an exact configured name, excluding frontend and reserved orchestration tools."""
    return tool.kind != "external" and tool.name not in LOCAL_TOOL_NAMES and tool.name in bypass


def is_capability_tool(ctx: RunContext[Any], tool: ToolDefinition) -> bool:
    """Trust tools supplied by an active, framework-registered capability.

    Core's MCP metadata alone cannot establish capability ownership, and frontend tools remain
    excluded even if they use a capability tool's name.
    """
    return (
        tool.kind != "external"
        and tool.name not in LOCAL_TOOL_NAMES
        and tool.capability_id is not None
        and tool.capability_id in ctx.capabilities
        and tool.capability_id in ctx.active_capability_ids
    )
