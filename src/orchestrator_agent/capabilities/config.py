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

"""Capability specs for the WFO agent.

Each capability is declared as a ``CapabilitySpec`` (pure data: id, description,
instructions, tags). The full MCP toolset is passed directly to the Agent — the MCP
server (via AgentTag.EXPOSED) is the single gate on what the model can call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from orchestrator_agent.capabilities.prompts import (
    get_aggregation_instructions,
    get_entity_instructions,
    get_export_instructions,
    get_search_instructions,
)


class CapabilitySpec(BaseModel):
    """Declarative definition of one capability, projected to a pydantic-ai ``Capability``."""

    id: str = Field(description="Stable capability id; the model loads it by this id.")
    description: str = Field(
        description="One-line catalog entry shown to the model (and reusable as the A2A skill description)."
    )
    instructions: str = Field(description="Instructions loaded with the capability when the model activates it.")
    defer_loading: bool = Field(
        default=True,
        description="Load on demand via load_capability (True) or keep always-on (False, e.g. shared helpers).",
    )
    advertise: bool = Field(
        default=True,
        description="Advertise this capability as an A2A AgentCard skill (False for internal helpers).",
    )
    a2a_tags: list[str] = Field(
        default_factory=list,
        description="Tags for the derived A2A skill (defaults to [id] when empty).",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Example prompts for this capability, surfaced in the A2A AgentCard skill.",
    )


def default_capability_specs() -> list[CapabilitySpec]:
    """The built-in WFO capability set."""
    return [
        CapabilitySpec(
            id="search",
            description="Find subscriptions, products, workflows, processes with lenient text/range filters.",
            instructions=get_search_instructions(),
            defer_loading=False,
            a2a_tags=["search", "query", "fuzzy", "semantic"],
            examples=["Find all active subscriptions", "Search for workflows containing 'migrate'"],
        ),
        CapabilitySpec(
            id="aggregate",
            description="Count, sum, average with grouping (regular or temporal).",
            instructions=get_aggregation_instructions(),
            defer_loading=False,
            a2a_tags=["aggregate", "analytics"],
            examples=["How many subscriptions per status?", "Count processes grouped by type"],
        ),
        CapabilitySpec(
            id="entity",
            description="Resolve a single entity by id or id-prefix.",
            instructions=get_entity_instructions(),
            defer_loading=False,
            a2a_tags=["details", "lookup"],
            examples=["Show details for subscription abc123", "Look up workflow with id prefix 4f2e"],
        ),
        CapabilitySpec(
            id="export",
            description="Prepare a downloadable export of an existing query's results.",
            instructions=get_export_instructions(),
            defer_loading=False,
            a2a_tags=["export"],
            examples=["Export the last search results as CSV"],
        ),
    ]


def load_capability_specs() -> list[CapabilitySpec]:
    """Return the active capability specs."""
    return default_capability_specs()


def skills_from_specs() -> list:
    """Project advertised capability specs into A2A AgentSkill objects."""
    from a2a.types import AgentSkill

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
        for spec in load_capability_specs()
        if spec.advertise
    ]


__all__ = [
    "CapabilitySpec",
    "load_capability_specs",
    "skills_from_specs",
]
