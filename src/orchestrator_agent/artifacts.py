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

from typing import Any, Literal

from orchestrator.search.query.results import VisualizationType
from pydantic import BaseModel, Field

FormFieldType = Literal["text", "int", "float", "bool", "select", "multiselect", "date", "uuid"]


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


class FormField(BaseModel):
    """One field in a workflow form, derived from a JSON-Schema property."""

    name: str
    label: str
    type: FormFieldType
    required: bool = True
    value: Any | None = None
    options: list[dict[str, str]] | None = None
    help_text: str | None = None
    error: str | None = None
    suggested_value: Any | None = None


class RenderFormArtifact(ToolArtifact):
    """A form page to be rendered by a surface adapter (Slack Block Kit, etc.)."""

    form_id: str
    title: str
    fields: list[FormField]
    submit_label: str = "Submit"
    cancel_label: str | None = "Cancel"


class ConfirmRequestArtifact(ToolArtifact):
    """A confirmation prompt summarising accumulated form pages before commit."""

    request_id: str
    title: str
    summary: str | dict[str, Any]
    confirm_label: str = "Confirm"
    cancel_label: str = "Cancel"
