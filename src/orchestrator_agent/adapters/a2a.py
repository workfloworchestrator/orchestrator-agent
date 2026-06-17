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

"""A2A adapter — exposes the orchestrator agent via the A2A protocol.

Uses the ``a2a-sdk`` server primitives (``AgentExecutor``, ``DefaultRequestHandler``,
``A2AFastAPIApplication``). The executor drives the plain capabilities-based agent
inside ``async with agent:`` (MCP session) and collects artifact results + final text.

``A2A_SKILLS`` is static A2A protocol metadata (the agent's advertised skills) and is
unrelated to pydantic-ai capabilities — it just describes what the agent can do.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    DataPart,
    Part,
    TextPart,
)
from fastapi import FastAPI
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.agent import new_deps
from orchestrator_agent.artifacts import QueryArtifact, ToolArtifact
from orchestrator_agent.capabilities.spec import skills_from_specs
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.persistence import PostgresStatePersistence

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)

NO_RESULTS = "No results"


A2A_SKILLS = skills_from_specs()


class WFOAgentExecutor(AgentExecutor):
    """AgentExecutor that drives the capabilities-based agent's event stream.

    Consumes the pydantic-ai event stream and publishes A2A events
    (status updates, artifacts) via the ``TaskUpdater`` helper.
    """

    def __init__(self, agent: "WFOAgent") -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.start_work()

        user_input = context.get_user_input()
        auth_token = self._parse_auth_token(context.message) if context.message else None

        deps = new_deps(user_input=user_input)

        from orchestrator.core.db import db
        from orchestrator.core.db.models import AgentRunTable

        try:
            deps.state.run_id = uuid.uuid4()
            agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=context_id, agent_type="a2a")
            db.session.add(agent_run)
            db.session.commit()

            logger.debug("A2A execute: starting", task_id=task_id, user_input=user_input[:100] if user_input else "")

            # Multi-turn memory: load the prior conversation for this context_id and replay it as
            # pydantic-ai message_history, so the proxy only needs to send the latest user turn.
            persistence = PostgresStatePersistence(thread_id=context_id, run_id=deps.state.run_id, session=db.session)
            prior_state = await persistence.load_state()
            message_history = (
                ModelMessagesTypeAdapter.validate_python(prior_state.message_history)
                if prior_state and prior_state.message_history
                else None
            )

            final_output = ""
            injected_blocks: list[str] = []

            with bind_outbound_token(auth_token):
                async with self.agent:
                    async with self.agent.run_stream_events(
                        user_input, deps=deps, message_history=message_history
                    ) as event_stream:
                        async for event in event_stream:
                            if isinstance(event, FunctionToolResultEvent):
                                result = event.part
                                if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
                                    data = json.loads(result.model_response_str())
                                    await updater.add_artifact(parts=[Part(root=DataPart(data=data))])
                                    # Collect chart/table blocks to append after the agent's prose.
                                    # The agent only sees raw data (no ToolReturn.content relay), so
                                    # it just summarises — the adapter guarantees the block appears.
                                    if isinstance(result.metadata, QueryArtifact) and result.metadata.rendered_block:
                                        injected_blocks.append(result.metadata.rendered_block.to_markdown())
                            elif isinstance(event, AgentRunResultEvent):
                                final_output = str(event.result.output)
                                deps.state.message_history = ModelMessagesTypeAdapter.dump_python(
                                    event.result.all_messages(), mode="json"
                                )
                            elif not isinstance(event, PartDeltaEvent):
                                logger.debug("A2A execute: event", event_type=type(event).__name__)

            if injected_blocks:
                final_output = final_output.rstrip() + "\n\n" + "\n\n".join(injected_blocks)

            await persistence.snapshot(deps.state)
            db.session.commit()

            # The answer is the agent's prose + any deterministically rendered chart/table blocks.
            await updater.complete(
                message=updater.new_agent_message(
                    parts=[Part(root=TextPart(text=final_output or NO_RESULTS))],
                )
            )

        except Exception:
            db.session.rollback()
            logger.exception("A2A execute: Task failed", task_id=context.task_id)
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[Part(root=TextPart(text="Task execution failed"))],
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.cancel()

    @staticmethod
    def _parse_auth_token(message: Any) -> str | None:
        """Extract a bearer token from message metadata, if any (for MCP forwarding)."""
        metadata = getattr(message, "metadata", None) or {}
        token = metadata.get("auth_token") or metadata.get("authToken")
        return str(token) if token else None


class A2AAdapter:
    """Wires the A2A protocol layer and adds routes to a FastAPI app.

    Usage::

        adapter = A2AAdapter(agent, url="http://localhost:8080/")
        adapter.add_routes(app)
    """

    def __init__(self, agent: "WFOAgent", url: str = "") -> None:
        self.agent = agent
        self.executor = WFOAgentExecutor(agent)

        agent_card = AgentCard(
            name="WFO Agent",
            description="Search, filter and aggregate orchestration data",
            url=url,
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=A2A_SKILLS,
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
        )

        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=task_store,
        )

        self._a2a_app = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

    def add_routes(self, app: FastAPI) -> None:
        """Add A2A protocol routes to an existing FastAPI application."""
        self._a2a_app.add_routes_to_app(app)
