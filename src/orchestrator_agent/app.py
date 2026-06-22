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

"""Standalone FastAPI app for the WFO search agent."""

from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from orchestrator.core.db import init_database

from orchestrator_agent.adapters import A2AAdapter, MCPApp
from orchestrator_agent.agent import build_agent
from orchestrator_agent.api.api import api_router
from orchestrator_agent.mcp_client import build_core_toolset, verify_tool_contract
from orchestrator_agent.security import AuthMiddleware, create_auth_manager
from orchestrator_agent.settings import agent_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: DB init, migration, adapter startup/shutdown."""
    init_database(agent_settings)  # type: ignore[arg-type]  # AgentSettings has DATABASE_URI which is all init_database needs

    # Fail loud at startup if the live MCP server's tools have drifted from what the agent expects.
    await verify_tool_contract(build_core_toolset())

    a2a_url = f"{agent_settings.BASE_URL}/"
    a2a = A2AAdapter(build_agent(agent_settings.create_model()), url=a2a_url)
    a2a.add_routes(app)

    mcp_app = MCPApp(build_agent(agent_settings.create_model()))
    app.mount("/mcp", mcp_app.app)

    # Manage adapter lifecycles. The agents open MCP sessions per-run (async with agent),
    # so only the MCP server's session manager needs lifespan management here.
    stack = AsyncExitStack()
    await stack.__aenter__()
    await stack.enter_async_context(mcp_app)

    logger.info(
        "Agent adapters started",
        a2a_url=a2a_url,
        mcp_path="/mcp",
        agent_model=agent_settings.AGENT_MODEL,
    )

    yield

    await stack.__aexit__(None, None, None)
    logger.info("Agent adapters stopped")


app = FastAPI(title="WFO Search Agent", lifespan=lifespan)
app.include_router(api_router)
app.add_middleware(AuthMiddleware, auth_manager=create_auth_manager())
