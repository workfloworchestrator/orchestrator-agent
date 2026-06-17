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

"""Artifact builders: map a tool's raw result payload to a rich artifact.

Each builder follows the shared protocol ``(tool_name, payload) -> ToolArtifact | None``. They are
the reusable, testable units; a plugin selects one via its frontmatter ``artifact:`` declaration
(see ``behavior/__init__.py``), and ``PluginCapability`` calls it (handling ownership filtering,
error isolation, and ToolReturn wrapping).
"""

from __future__ import annotations

from typing import Any, get_args

from orchestrator.core.search.query.results import VisualizationType

from orchestrator_agent.artifacts import (
    DataArtifact,
    ExportArtifact,
    QueryArtifact,
    RenderedBlock,
)
from orchestrator_agent.rendering.charts import aggregate_to_mermaid
from orchestrator_agent.rendering.tables import search_to_markdown
from orchestrator_agent.tool_names import AGGREGATE_TOOL, SEARCH_TOOL

# --- Shared artifact builders (raw MCP payload -> artifact, or None) ------------------------


def query_artifact(tool_name: str, payload: Any) -> QueryArtifact | None:
    """Build a QueryArtifact for a search/aggregate payload, upgraded with a chart or table block."""
    if not isinstance(payload, dict):
        return None
    query = _base_query_artifact(tool_name, payload)
    if query is None:
        return None
    if tool_name == AGGREGATE_TOOL and (mermaid := aggregate_to_mermaid(payload)):
        return query.model_copy(update={"rendered_block": RenderedBlock(type="mermaid", content=mermaid)})
    if tool_name == SEARCH_TOOL and (markdown := search_to_markdown(payload)):
        return query.model_copy(update={"rendered_block": RenderedBlock(type="markdown", content=markdown)})
    return query


def _base_query_artifact(tool_name: str, payload: dict[str, Any]) -> QueryArtifact | None:
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
    # Only honour a visualization the model can render; an out-of-range value would raise a
    # ValidationError that the builder's error isolation swallows, silently dropping the artifact.
    if (visualization := payload.get("visualization")) in get_args(VisualizationType.model_fields["type"].annotation):
        kwargs["visualization_type"] = {"type": visualization}
    if search_type := payload.get("search_type"):
        kwargs["search_type"] = search_type
    return QueryArtifact(**kwargs)


def data_artifact(tool_name: str, payload: Any) -> DataArtifact | None:
    """Build a DataArtifact for a resolve_entity payload (a concrete entity), or None.

    ``tool_name`` is part of the shared builder protocol (``(tool_name, payload) -> artifact``) but
    unused here — entity results map the same way regardless of which tool produced them.
    """
    if not isinstance(payload, dict):
        return None
    entity_id = payload.get("entity_id") or payload.get("subscription_id") or payload.get("id")
    entity_type = payload.get("entity_type")
    if entity_id is None or entity_type is None:
        return None
    return DataArtifact(
        description=f"Fetched details for {entity_type} {entity_id}",
        entity_id=str(entity_id),
        entity_type=str(entity_type),
    )


def export_artifact(tool_name: str, payload: Any) -> ExportArtifact | None:
    """Build an ExportArtifact for an export_query payload (a prepared download), or None.

    ``tool_name`` is part of the shared builder protocol but unused here.
    """
    if not isinstance(payload, dict):
        return None
    download = payload.get("download_path") or payload.get("download_url")
    if not download:
        return None
    return ExportArtifact(
        description="Prepared export for download",
        query_id=str(payload.get("query_id", "")),
        download_url=str(download),
    )


__all__ = [
    "data_artifact",
    "export_artifact",
    "query_artifact",
]
