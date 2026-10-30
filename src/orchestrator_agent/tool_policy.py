"""Read-only Core discovery and explicit mutation access through registered capabilities."""

from collections.abc import Collection
from dataclasses import replace
from typing import Any, TypeVar

from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL, WRITE_TOOL_NAMES

LOCAL_TOOL_NAMES = frozenset({START_WORKFLOW_FORM_TOOL, "load_capability"})
_CAPABILITY_MUTATION_KEY = "wfo_capability_mutation"
DepsT = TypeVar("DepsT")


def is_read_only_core_tool(tool: ToolDefinition) -> bool:
    """Require an explicit boolean read-only declaration, keeping writes and local names out.

    Core authors are responsible for accurate annotations. The declaration controls admission;
    it does not inspect the endpoint implementation or prove that it cannot change business state.
    """
    if tool.kind == "external" or tool.name in WRITE_TOOL_NAMES or tool.name in LOCAL_TOOL_NAMES:
        return False
    annotations = (tool.metadata or {}).get("annotations")
    return isinstance(annotations, dict) and annotations.get("readOnlyHint") is True


def allow_capability_mutations(
    toolset: AbstractToolset[DepsT], *, tool_names: Collection[str]
) -> AbstractToolset[DepsT]:
    """Expose only the named tools when returned by a registered capability's ``get_toolset``.

    The toolset may contain local functions or Core MCP writes. This is an explicit bypass of the
    default read-only policy, not an approval mechanism: the capability must enforce any required
    authorization and approval. Returning this as an ordinary agent/run toolset grants no bypass.
    """
    names = frozenset(tool_names)
    if not names:
        raise ValueError("Select at least one tool for the capability mutation bypass")
    if reserved := names & LOCAL_TOOL_NAMES:
        raise ValueError(f"Local orchestration names cannot be bypassed: {sorted(reserved)}")

    async def prepare(_ctx: RunContext[DepsT], tools: list[ToolDefinition]) -> list[ToolDefinition]:
        return [
            replace(tool, metadata={**(tool.metadata or {}), _CAPABILITY_MUTATION_KEY: True})
            for tool in tools
            if tool.name in names
        ]

    return toolset.prepared(prepare)


def is_capability_mutation_tool(ctx: RunContext[Any], tool: ToolDefinition) -> bool:
    """Require both explicit opt-in and an active, framework-registered capability owner.

    Core's MCP metadata alone cannot establish capability ownership, and frontend tools remain
    excluded even if they use a selected tool's name.
    """
    return (
        tool.kind != "external"
        and tool.name not in LOCAL_TOOL_NAMES
        and (tool.metadata or {}).get(_CAPABILITY_MUTATION_KEY) is True
        and tool.capability_id is not None
        and tool.capability_id in ctx.capabilities
        and tool.capability_id in ctx.active_capability_ids
    )
