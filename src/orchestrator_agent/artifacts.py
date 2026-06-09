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

from typing import Literal

from orchestrator.core.search.query.results import VisualizationType
from pydantic import BaseModel, Field


class RenderedBlock(BaseModel):
    """A pre-rendered output block ready to inject into the agent's text reply."""

    type: Literal["mermaid", "markdown"]
    content: str

    def to_markdown(self) -> str:
        """Render as injectable Markdown.

        Mermaid gets a code fence; Markdown passes through raw (a fenced table would render as
        literal source instead of a table).
        """
        if self.type == "mermaid":
            return f"```mermaid\n{self.content}\n```"
        return self.content


class ToolArtifact(BaseModel):
    """Base metadata for result-producing tools.

    Adapters check ``isinstance(metadata, ToolArtifact)`` to distinguish
    result-producing tools from intermediate setup tools.
    """

    description: str


class QueryArtifact(ToolArtifact):
    """Lightweight reference returned by query tools."""

    query_id: str
    total_results: int
    visualization_type: VisualizationType = Field(default_factory=VisualizationType)
    search_type: str = ""
    rendered_block: RenderedBlock | None = None


class DataArtifact(ToolArtifact):
    """Metadata for tools that return full entity data for LLM reasoning."""

    entity_id: str
    entity_type: str


class ExportArtifact(ToolArtifact):
    """Metadata for tools that produce a downloadable export reference."""

    query_id: str
    download_url: str
