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

"""The WFO agent — a plain pydantic-ai ``Agent`` configured with capabilities.

Domain behaviour lives entirely in capabilities (see ``capabilities.py``): each
wraps a slice of orchestrator-core's MCP toolset plus its instructions, and is
loaded on demand by the model via the framework ``load_capability`` tool.

MCP tools require an open session, so every run must happen inside
``async with agent:``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps

from orchestrator_agent.capabilities.hooks import build_capabilities
from orchestrator_agent.capabilities.loader import load_system_prompt
from orchestrator_agent.mcp_client import build_core_toolset
from orchestrator_agent.state import SearchState

if TYPE_CHECKING:
    from pydantic_ai.models import KnownModelName, Model

logger = structlog.get_logger(__name__)

WFOAgent = Agent[StateDeps[SearchState], str]


def build_agent(model: "Model | KnownModelName | str") -> WFOAgent:
    """Build the capabilities-based WFO agent.

    Args:
        model: A pydantic-ai model or model name/string.

    Returns:
        A plain ``Agent`` ready to run inside ``async with agent:``.
    """
    capabilities = build_capabilities()
    logger.debug("Building WFO agent", model=str(model), capability_count=len(capabilities))
    agent: WFOAgent = Agent(
        model=model,
        deps_type=StateDeps[SearchState],
        instructions=load_system_prompt(),
        toolsets=[build_core_toolset()],
        capabilities=capabilities,
        retries=2,
    )
    return agent


def new_deps(user_input: str = "") -> StateDeps[SearchState]:
    """Convenience constructor for the agent's run dependencies."""
    return StateDeps(SearchState(user_input=user_input))


__all__: list[str] = ["WFOAgent", "build_agent", "new_deps"]
