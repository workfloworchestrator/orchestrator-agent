# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native pydantic-ai capabilities for the WFO agent.

Each domain capability wraps a slice of orchestrator-core's MCP toolset, filtered
to the tools it owns, plus the instructions ported from the old per-skill prompts.
``defer_loading=True`` lets the model load a capability on demand via the framework
``load_capability`` tool — this replaces the old planner's routing.

``ArtifactCapability`` is an agent-level capability whose ``after_tool_execute``
hook maps each MCP tool's structured JSON result into the ``QueryArtifact`` /
``DataArtifact`` / ``ExportArtifact`` types via ``ToolReturn.metadata``. The AG-UI
and text adapters already key off ``isinstance(result.metadata, ToolArtifact)``,
so this keeps the rich-artifact path working now that tools come from MCP.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic_ai.capabilities import AbstractCapability, Capability, ProcessHistory
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturn
from pydantic_ai.tools import RunContext, ToolDefinition

from orchestrator_agent.artifacts import (
    DataArtifact,
    ExportArtifact,
    QueryArtifact,
    RenderedBlock,
)
from orchestrator_agent.capabilities.config import CapabilitySpec, load_capability_specs
from orchestrator_agent.rendering.charts import aggregate_to_mermaid
from orchestrator_agent.rendering.tables import search_to_markdown
from orchestrator_agent.tool_names import (
    AGGREGATE_TOOL,
    DISCOVER_FILTER_PATHS_TOOL,
    EXPORT_QUERY_TOOL,
    GET_ENTITY_DETAILS_TOOL,
    PATH_CONSUMING_PARAMS,
    RESOLVE_ENTITY_TOOL,
    SEARCH_TOOL,
)

logger = structlog.get_logger(__name__)


def build_capabilities(specs: list[CapabilitySpec] | None = None) -> list[AbstractCapability[Any]]:
    """Build the agent's capability list from specs (defaulting to the configured set).

    Capabilities provide instructions and hooks only — no tool filtering. The full MCP
    toolset is passed directly to the Agent so the MCP server (via AgentTag.EXPOSED) remains
    the single gate on what the model can call. New tools on the server are automatically
    available without any changes here.
    """
    resolved = specs if specs is not None else load_capability_specs()
    domain_caps: list[AbstractCapability[Any]] = [
        Capability[Any](
            id=spec.id,
            description=spec.description,
            instructions=spec.instructions,
            defer_loading=spec.defer_loading,
        )
        for spec in resolved
    ]
    return [
        *domain_caps,
        FilterPathGuard(),
        ProcessHistory[Any](processor=trim_history),
        ArtifactCapability(),
    ]


# --- Deterministic search flow (discover filter paths before searching) --------------------


class FilterPathGuard(AbstractCapability[Any]):
    """Enforce the discover-before-filter ordering that makes search deterministic.

    Filter and ``group_by`` paths are database-specific — the model cannot guess them and
    must read them from ``discover_filter_paths`` first. Rather than trust prompt discipline
    (the old failure mode: invented paths, context bleed), this ``before_tool_execute`` hook
    makes the ordering a hard precondition: when the model calls *any* tool that supplies a
    path-consuming argument without having called the discovery tool earlier in the run, it
    raises ``ModelRetry`` to force the ``discover_filter_paths`` -> ``search``/``aggregate``
    sequence.

    Which arguments count as "path-consuming" is detected from each tool's *own* JSON schema
    (the names in ``PATH_CONSUMING_PARAMS``), so there is no hardcoded ``search``/``aggregate``
    tool-name list — any future tool that accepts ``filters``/``group_by`` is gated the same way.
    """

    def __init__(
        self,
        discovery_tool: str = DISCOVER_FILTER_PATHS_TOOL,
        path_params: tuple[str, ...] = PATH_CONSUMING_PARAMS,
    ) -> None:
        self.discovery_tool = discovery_tool
        self.path_params = path_params

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (behaviour, not data).

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        schema_props = (tool_def.parameters_json_schema or {}).get("properties", {})
        used = [p for p in self.path_params if p in schema_props and args.get(p)]
        if not used or self._discovery_called(ctx):
            return args
        raise ModelRetry(
            f"'{tool_def.name}' was called with {used}, but filter/group-by paths are "
            f"database-specific and must not be guessed. Call '{self.discovery_tool}' first to "
            f"obtain valid paths for those fields, then retry '{tool_def.name}'."
        )

    def _discovery_called(self, ctx: RunContext[Any]) -> bool:
        """True if the discovery tool was already called earlier in this run."""
        return any(
            isinstance(part, ToolCallPart) and part.tool_name == self.discovery_tool
            for message in ctx.messages
            for part in getattr(message, "parts", [])
        )


# --- History processing (replaces the old per-skill memory_scope trimming) -----------------

# Keep only the most recent turns on the wire. The old MemoryScope machinery trimmed
# per-skill execution traces; with a single agent + MCP we keep a sliding window of the
# raw conversation instead, which is what the model actually needs to follow context.
_MAX_HISTORY_MESSAGES = 30


def trim_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Cap message history to a sliding window, preserving leading instructions/system parts.

    Keeps any leading request that carries instructions (so the system framing stays in
    place) plus the most recent ``_MAX_HISTORY_MESSAGES`` messages.
    """
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        return messages

    head: list[ModelMessage] = []
    rest = messages
    first = messages[0]
    if isinstance(first, ModelRequest) and getattr(first, "instructions", None):
        head = [first]
        rest = messages[1:]

    tail = rest[-_MAX_HISTORY_MESSAGES:]
    return head + tail


# --- Artifact mapping (MCP JSON result -> rich artifact metadata) --------------------------


class ArtifactCapability(AbstractCapability[Any]):
    """Maps MCP tool JSON results into rich artifact metadata for the AG-UI/text paths.

    The MCP tools return plain structured JSON. Adapters identify consumer-facing results
    by ``isinstance(ToolReturnPart.metadata, ToolArtifact)``; this hook re-wraps the
    relevant tool results in a ``ToolReturn`` carrying the matching artifact so that
    behaviour is preserved without the old local tools.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable.

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        result: Any,
    ) -> Any:
        try:
            artifact = _artifact_for(tool_def.name, result)
        except Exception:  # noqa: BLE001 - artifact mapping must never break tool execution
            logger.exception("Artifact mapping failed; returning raw result", tool=tool_def.name)
            return result
        if artifact is None:
            return result

        # When we render a chart/table the adapter appends it after the agent's reply. The agent
        # never sees that block, so tell it — only on the calls we actually inject — to summarise
        # rather than restate the data (otherwise it lists the rows and the output duplicates).
        notice = _injection_notice(artifact)
        if isinstance(result, ToolReturn):
            result.metadata = artifact
            if notice and result.content is None:
                result.content = notice
            return result
        return ToolReturn(return_value=result, metadata=artifact, content=notice)


def _result_payload(result: Any) -> Any:
    """Unwrap a ToolReturn to its underlying value; pass other results through."""
    return result.return_value if isinstance(result, ToolReturn) else result


def _injection_notice(artifact: Any) -> str | None:
    """A note for the agent when this result carries a chart/table the adapter will append, else None.

    Goes in ``ToolReturn.content`` (which the agent sees) — never the block itself, so the agent
    can't reproduce or mangle it. Only emitted when there is actually something to inject.
    """
    block = getattr(artifact, "rendered_block", None)
    if block is None:
        return None
    kind = "chart" if block.type == "mermaid" else "table"
    return (
        f"A {kind} for this result is shown to the user automatically below your reply. "
        "Summarise in one sentence; do not reproduce it or restate its rows/numbers."
    )


def _build_query_artifact(tool_name: str, payload: dict[str, Any]) -> QueryArtifact | None:
    query_id = payload.get("query_id")
    if query_id is None:
        return None
    total = payload.get("returned") or payload.get("total_results")
    if total is None:
        results = payload.get("results")
        total = len(results) if isinstance(results, list) else 0
    kwargs: dict[str, Any] = {
        "description": f"{tool_name} returned {total} result(s)",
        "query_id": str(query_id),
        "total_results": int(total),
    }
    if visualization := payload.get("visualization"):
        kwargs["visualization_type"] = {"type": visualization}
    if search_type := payload.get("search_type"):
        kwargs["search_type"] = search_type
    return QueryArtifact(**kwargs)


def _artifact_for(tool_name: str, result: Any) -> QueryArtifact | DataArtifact | ExportArtifact | None:
    """Build the artifact for a known MCP tool result, or None for setup/unknown tools."""
    payload = _result_payload(result)
    if not isinstance(payload, dict):
        return None

    if tool_name in (SEARCH_TOOL, AGGREGATE_TOOL):
        query = _build_query_artifact(tool_name, payload)
        if query is None:
            return None
        if tool_name == AGGREGATE_TOOL and (mermaid := aggregate_to_mermaid(payload)):
            return query.model_copy(update={"rendered_block": RenderedBlock(type="mermaid", content=mermaid)})
        if tool_name == SEARCH_TOOL and (markdown := search_to_markdown(payload)):
            return query.model_copy(update={"rendered_block": RenderedBlock(type="markdown", content=markdown)})
        return query

    if tool_name == EXPORT_QUERY_TOOL:
        download = payload.get("download_path") or payload.get("download_url")
        if not download:
            return None
        return ExportArtifact(
            description="Prepared export for download",
            query_id=str(payload.get("query_id", "")),
            download_url=str(download),
        )

    if tool_name in (GET_ENTITY_DETAILS_TOOL, RESOLVE_ENTITY_TOOL):
        entity_id = payload.get("entity_id") or payload.get("subscription_id") or payload.get("id")
        entity_type = payload.get("entity_type")
        if entity_id is None or entity_type is None:
            return None
        return DataArtifact(
            description=f"Fetched details for {entity_type} {entity_id}",
            entity_id=str(entity_id),
            entity_type=str(entity_type),
        )

    return None


__all__ = [
    "ArtifactCapability",
    "FilterPathGuard",
    "build_capabilities",
    "trim_history",
]
