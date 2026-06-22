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

"""Plugin model for the WFO agent.

A *plugin* is a single authored Markdown file with YAML frontmatter (``plugins/<id>.md``), loaded by
``loader.load_plugin_specs``: ``PluginSpec`` validates the frontmatter and carries the body verbatim
as ``instructions``.

The spec's fields project onto a pydantic-ai ``Capability`` (``id``/``description``/``instructions``/
``defer_loading``, via ``behavior.build_plugin_capability``) and, when advertised, an A2A
``AgentSkill``. The full MCP toolset is passed directly to the Agent — the MCP server (via
AgentTag.EXPOSED) is the single gate on what the model can call.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ArtifactType(StrEnum):
    """The result-mapping a plugin declares via frontmatter ``artifact:``.

    Each value is bound to a builder in ``behavior.ARTIFACT_BUILDERS``; a typed field gives free
    validation (a bad ``artifact:`` fails at load) and keeps the strings defined in exactly one place.
    """

    QUERY = "query"  # search / aggregate -> QueryArtifact (+ Mermaid chart or Markdown table)
    DATA = "data"  # entity -> DataArtifact (a fetched domain object)
    EXPORT = "export"  # export -> ExportArtifact (a prepared download)


class PluginSpec(BaseModel):
    """A loaded plugin: validated frontmatter plus the Markdown body as ``instructions``.

    ``extra='forbid'`` so a mistyped frontmatter key fails loudly. ``instructions`` is filled from
    the body by the loader, not the frontmatter.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable plugin id; also the projected Capability/Skill id.")
    description: str = Field(description="One-line catalog entry shown to the model and as the A2A skill description.")
    instructions: str = Field(default="", description="The Markdown body; filled by the loader, not the frontmatter.")
    advertise: bool = Field(default=True, description="Project this plugin to an A2A AgentCard skill.")
    a2a_tags: list[str] = Field(default_factory=list, description="Tags for the derived A2A skill (defaults to [id]).")
    examples: list[str] = Field(
        default_factory=list, description="Example prompts surfaced in the A2A AgentCard skill."
    )
    defer_loading: bool = Field(
        description="Required. If True, the capability's tools/instructions are hidden until the model "
        "loads it on demand (pydantic-ai load_capability). False = always-on. Keep False unless the "
        "capability set grows large enough that on-demand loading is worth the routing. Required (no "
        "default) so every plugin states its loading mode explicitly.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool-name constants from tool_names.py this plugin OWNS (e.g. SEARCH_TOOL). Ownership "
        "drives artifact mapping (its results -> the declared artifact) and DeferredToolGate (these tools "
        "hide with the plugin when deferred). Prompts do not name tools; the model binds intent from the "
        "tool's description.",
    )
    artifact: ArtifactType | None = Field(
        default=None,
        description="Which shared artifact mapping this plugin's tool results use (query/data/export). "
        "Omit for an instructions-only plugin.",
    )


def skills_from_specs() -> list:
    """Project advertised plugin specs into A2A AgentSkill objects."""
    from a2a.types import AgentSkill

    from orchestrator_agent.capabilities.loader import load_plugin_specs

    return [
        AgentSkill(
            id=spec.id,
            name=spec.id.replace("_", " ").title(),
            description=spec.description,
            tags=spec.a2a_tags or [spec.id],
            examples=spec.examples or None,
            input_modes=["application/json"],
            output_modes=["text/markdown", "application/json"],
        )
        for spec in load_plugin_specs()
        if spec.advertise
    ]


__all__ = [
    "ArtifactType",
    "PluginSpec",
    "skills_from_specs",
]
