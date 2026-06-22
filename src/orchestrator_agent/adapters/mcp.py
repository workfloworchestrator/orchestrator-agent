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

This adapter exposes the agent *itself* as an MCP server (distinct from the agent being an MCP
*client* of orchestrator-core). Each advertised plugin is auto-registered as one MCP tool — its name
and description come from the plugin spec — so the MCP surface stays in sync with the plugins with no
per-tool wiring here: add a plugin, it appears as a tool. A generic ``ask`` catch-all is also
exposed.

Every tool is a thin natural-language entry point: it runs the full agent (always-on capabilities,
which route by intent) via ``MCPWorker`` and returns the combined result string. The tool name/
description simply advertise the capability to an MCP client browsing the agent's tools. This
adapter is stateless — each tool call is independent.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import structlog
from mcp.server.fastmcp import FastMCP
from orchestrator.core.db import db
from orchestrator.core.db.models import AgentRunTable
from pydantic_ai.messages import ToolReturnPart

from orchestrator_agent.agent import new_deps
from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.capabilities import load_plugin_specs
from orchestrator_agent.mcp_client import bind_outbound_token

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent
    from orchestrator_agent.capabilities import PluginSpec

logger = structlog.get_logger(__name__)


def _tool_description(spec: "PluginSpec") -> str:
    """Build an MCP tool description from a plugin spec (its one-liner + examples + an NL hint)."""
    parts = [spec.description]
    if spec.examples:
        parts.append("Examples: " + "; ".join(f'"{e}"' for e in spec.examples))
    parts.append("Describe your request in natural language.")
    return "\n\n".join(parts)


def _result_with_data(result: Any) -> str:
    """The agent's final answer, plus the structured tool payloads as JSON.

    The agent's prose is a one-line takeaway — it's told the data is rendered elsewhere (a chart/table
    for human clients). An MCP caller is an LLM, so we send that data too, but as its native JSON
    (not mermaid/markdown): the raw payloads of the artifact-bearing tool results from this run.
    """
    data = [
        part.model_response_str()
        for msg in result.all_messages()
        for part in getattr(msg, "parts", [])
        if isinstance(part, ToolReturnPart) and isinstance(part.metadata, ToolArtifact)
    ]
    return "\n\n".join([result.output, *data]) if data else result.output


class MCPWorker:
    """Runs the agent for a natural-language query and returns a result string."""

    def __init__(self, agent: "WFOAgent") -> None:
        self.agent = agent

    async def run(self, query: str, *, auth_token: str | None = None) -> str:
        """Run the agent (MCP session open) and return its final answer plus the structured data."""
        deps = new_deps(user_input=query)

        deps.state.run_id = uuid.uuid4()
        agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=str(uuid.uuid4()), agent_type="mcp")
        db.session.add(agent_run)
        db.session.commit()

        logger.debug("MCPWorker: Starting run", query=query[:100])

        try:
            with bind_outbound_token(auth_token):
                async with self.agent:
                    result = await self.agent.run(query, deps=deps)
            return _result_with_data(result)
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
        """Register one MCP tool per advertised plugin, plus a generic ``ask`` catch-all.

        The plugin set is the single source: each advertised ``PluginSpec`` becomes a tool named after
        the plugin, described by its spec. New plugins appear automatically; nothing is hardcoded here.
        """
        worker = self.worker

        for spec in load_plugin_specs():
            if spec.advertise:
                self.server.add_tool(self._run_query_tool(), name=spec.id, description=_tool_description(spec))

        @self.server.tool()
        async def ask(query: str) -> str:
            """Ask any question about the orchestration system in natural language.

            The agent picks the right capability (search, aggregate, entity lookup, export) and answers.
            """
            return await worker.run(query)

    def _run_query_tool(self) -> Callable[[str], Awaitable[str]]:
        """A fresh natural-language tool handler that runs the agent (one per registered plugin tool)."""
        worker = self.worker

        async def run_query(query: str) -> str:
            return await worker.run(query)

        return run_query

    async def __aenter__(self) -> MCPApp:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        await self._stack.enter_async_context(self.server.session_manager.run())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        await self._stack.__aexit__(exc_type, exc_val, exc_tb)
