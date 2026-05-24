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

"""Typed metadata for result-producing tools.

Adapters check ``isinstance(metadata, ToolArtifact)`` to uniformly identify
tool results that carry consumer-facing data (query results, exports, entity
details).  Setup tools (filters, grouping, etc.) return plain types and are
*not* wrapped in a ToolArtifact.
"""

from typing import Any

from orchestrator.search.query.results import VisualizationType
from pydantic import BaseModel, ConfigDict, Field


class ToolArtifact(BaseModel):
    """Base metadata for result-producing tools.

    Adapters check ``isinstance(metadata, ToolArtifact)`` to distinguish
    result-producing tools from intermediate setup tools.
    """

    description: str


class QueryArtifact(ToolArtifact):
    """Lightweight reference returned by query tools.

    Client fetches full results via GET /queries/{query_id}/results.
    """

    query_id: str
    total_results: int
    visualization_type: VisualizationType = Field(default_factory=VisualizationType)


class DataArtifact(ToolArtifact):
    """Metadata for tools that return full entity data for LLM reasoning."""

    entity_id: str
    entity_type: str


class ExportArtifact(ToolArtifact):
    """Metadata for tools that produce a downloadable export reference."""

    query_id: str
    download_url: str


class RenderFormArtifact(ToolArtifact):
    """A form page to be rendered by a schema-aware surface (LibreChat).

    ``form_schema`` (serialized as ``schema``) is the raw pydantic-forms JSON
    Schema returned by orchestrator-core's ``get_workflow_form`` MCP tool —
    forwarded verbatim so renderers (the ``pydantic-forms`` npm package,
    orchestrator-ui) consume it with full fidelity: conditional fields,
    nested objects, custom widget metadata, the lot.

    ``prefill`` carries LLM-extracted values keyed by field name (see
    :mod:`orchestrator_agent.tools.form_prefill`). Renderers should apply
    these as initial values without mutating the schema itself.

    Note on the alias: the Python attribute is named ``form_schema`` to
    avoid shadowing pydantic's deprecated ``BaseModel.schema()`` method,
    but the JSON field name is ``schema`` to match what the pydantic-forms
    renderer expects on the wire.
    """

    model_config = ConfigDict(populate_by_name=True)

    form_id: str
    title: str
    form_schema: dict[str, Any] = Field(alias="schema")
    prefill: dict[str, Any] | None = None
    submit_label: str = "Submit"
    cancel_label: str | None = "Cancel"


class ConfirmRequestArtifact(ToolArtifact):
    """A confirmation prompt summarising accumulated form pages before commit."""

    request_id: str
    title: str
    summary: str | dict[str, Any]
    confirm_label: str = "Confirm"
    cancel_label: str = "Cancel"
