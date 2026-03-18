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
``A2AFastAPIApplication``) instead of fasta2a.  The SDK handles all JSON-RPC routing,
SSE streaming, task lifecycle management, and agent card serving.
"""

from __future__ import annotations

import uuid
from typing import Any

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
    AgentSkill,
    Part,
    TextPart,
)
from fastapi import FastAPI
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.agent import AgentAdapter
from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.memory import FALLBACK_MESSAGE, collect_tool_descriptions
from orchestrator_agent.skills import SKILLS
from orchestrator_agent.state import SearchState, TaskAction

logger = structlog.get_logger(__name__)


def _build_state_fallback(state: SearchState) -> str:
    """Build A2A output from execution state when event stream yielded nothing useful."""
    completed = state.memory.completed_turns[-1:]
    if completed:
        return collect_tool_descriptions(completed[-1].steps)
    return FALLBACK_MESSAGE


A2A_SKILLS = [
    AgentSkill(
        id=action.value,
        name=skill.name,
        description=skill.description,
        tags=skill.tags,
        input_modes=["application/json"],
        output_modes=["application/json"],
    )
    for action, skill in SKILLS.items()
]


class WFOAgentExecutor(AgentExecutor):
    """AgentExecutor that drives ``AgentAdapter.run_stream_events()``.

    Consumes the pydantic-ai event stream and publishes A2A events
    (status updates, artifacts) via the ``TaskUpdater`` helper.
    """

    def __init__(self, agent: AgentAdapter) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.start_work()

        user_input = context.get_user_input()
        message = context.message
        target_action = self._parse_target_action(message) if message else None

        deps = StateDeps(SearchState(user_input=user_input))

        from orchestrator.db import db
        from orchestrator.db.models import AgentRunTable

        try:
            deps.state.run_id = uuid.uuid4()
            agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=context_id, agent_type="a2a")
            db.session.add(agent_run)
            db.session.commit()

            event_stream = self.agent.run_stream_events(deps=deps, target_action=target_action)
            artifact_texts: list[str] = []
            final_output = ""

            async for event in event_stream:
                logger.debug("A2A execute: event", event_type=type(event).__name__)

                if isinstance(event, FunctionToolResultEvent):
                    result = event.result
                    if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
                        text = result.model_response_str()
                        await updater.add_artifact(
                            parts=[Part(root=TextPart(text=text))],
                        )
                        artifact_texts.append(text)

                if isinstance(event, AgentRunResultEvent):
                    final_output = str(event.result.output)

            # Prefer artifact content over LLM text (matches collect_stream_output behavior)
            if artifact_texts:
                final_output = "\n\n".join(artifact_texts)
            elif not final_output or final_output in ("", FALLBACK_MESSAGE):
                final_output = _build_state_fallback(deps.state)

            await updater.complete(
                message=updater.new_agent_message(
                    parts=[Part(root=TextPart(text=final_output or "No results"))],
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
    def _parse_target_action(message: Any) -> TaskAction | None:
        """Extract target action from message metadata, if any."""
        metadata = getattr(message, "metadata", None) or {}
        skill_id = metadata.get("skill_id") or metadata.get("skillId")
        if not skill_id:
            return None
        try:
            action = TaskAction(skill_id)
            logger.debug("A2A: Routing to skill directly", target_action=action)
            return action
        except ValueError:
            logger.warning("A2A: Unknown skillId, falling back to planner", skill_id=skill_id)
            return None


class A2AAdapter:
    """Wires the A2A protocol layer and adds routes to a FastAPI app.

    Usage::

        adapter = A2AAdapter(agent, url="http://localhost:8000/")
        adapter.add_routes(app)
        # In lifespan:
        async with adapter.agent:
            yield
    """

    def __init__(self, agent: AgentAdapter, url: str = "") -> None:
        self.agent = agent
        self.executor = WFOAgentExecutor(agent)

        agent_card = AgentCard(
            name="WFO Search Agent",
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

    async def __aenter__(self) -> A2AAdapter:
        await self.agent.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.agent.__aexit__(*exc)
