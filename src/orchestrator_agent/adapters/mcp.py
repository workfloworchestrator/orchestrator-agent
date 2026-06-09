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

"""MCP (Model Context Protocol) adapter for the search agent.

``MCPWorker`` owns the agent lifecycle and stream consumption — parallel to
``AGUIWorker`` and the A2A executor. It runs the capabilities-based agent (which
self-routes via on-demand capability loading) and returns a combined result string.

The FastMCP tool handlers are thin natural-language entry points; the model
decides which domain capability to load. This adapter is stateless — each tool
call is independent.
"""

from __future__ import annotations

import uuid
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import structlog
from mcp.server.fastmcp import FastMCP
from orchestrator.core.db import db
from orchestrator.core.db.models import AgentRunTable
from orchestrator.core.search.core.types import EntityType

from orchestrator_agent.adapters.stream import collect_stream_output
from orchestrator_agent.agent import new_deps
from orchestrator_agent.mcp_client import bind_outbound_token

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)


class MCPWorker:
    """Runs the agent for a natural-language query and returns a result string."""

    def __init__(self, agent: "WFOAgent") -> None:
        self.agent = agent

    async def run(self, query: str, *, auth_token: str | None = None) -> str:
        """Create state, run the agent (MCP session open), and return a result string."""
        deps = new_deps(user_input=query)

        deps.state.run_id = uuid.uuid4()
        agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=str(uuid.uuid4()), agent_type="mcp")
        db.session.add(agent_run)
        db.session.commit()

        logger.debug("MCPWorker: Starting run", query=query[:100])

        try:
            with bind_outbound_token(auth_token):
                async with self.agent:
                    async with self.agent.run_stream_events(query, deps=deps) as event_stream:
                        output = await collect_stream_output(event_stream)
            return output
        except Exception:
            logger.exception("MCPWorker: run failed", query=query[:100])
            raise


class MCPApp:
    """MCP adapter app: FastMCP server, worker, tools, and lifecycle."""

    def __init__(self, agent: "WFOAgent") -> None:
        self.agent = agent
        self.worker = MCPWorker(agent)
        self._stack: AsyncExitStack

        self.server = FastMCP(
            name="WFO Search Agent",
            instructions="Search, filter and aggregate orchestration data",
            stateless_http=True,
            streamable_http_path="/",
        )
        self._register_tools()
        self.app = self.server.streamable_http_app()

    def _register_tools(self) -> None:
        worker = self.worker

        @self.server.tool()
        async def search(query: str) -> str:
            """Find subscriptions, products, workflows, or processes.

            Describe what you're looking for in natural language.
            Examples: "active subscriptions", "failed workflows from last week"
            """
            return await worker.run(query)

        @self.server.tool()
        async def aggregate(query: str) -> str:
            """Count, sum, or average data with grouping.

            Describe what aggregation you need.
            Examples: "count subscriptions by product", "average workflow duration by status"
            """
            return await worker.run(query)

        @self.server.tool()
        async def get_entity_details(entity_type: EntityType, entity_id: uuid.UUID) -> str:
            """Fetch full details for a specific entity.

            Args:
                entity_type: The type of entity (SUBSCRIPTION, PRODUCT, WORKFLOW, PROCESS)
                entity_id: The UUID of the entity
            """
            query = f"Get details for {entity_type.value} {entity_id}"
            return await worker.run(query)

        @self.server.tool()
        async def ask(query: str) -> str:
            """Ask any question about the orchestration system.

            The agent will determine the best approach — search, aggregate, or answer directly.
            """
            return await worker.run(query)

    async def __aenter__(self) -> MCPApp:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        await self._stack.enter_async_context(self.server.session_manager.run())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        await self._stack.__aexit__(exc_type, exc_val, exc_tb)
