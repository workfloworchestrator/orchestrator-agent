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


from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ag_ui.core import BaseEvent, EventType, RunAgentInput, ToolCallResultEvent
from orchestrator.core.db.models import AgentRunTable
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.ag_ui import AGUIEventStream as _BaseAGUIEventStream
from sqlalchemy.orm import Session
from structlog import get_logger

from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.persistence import PostgresStatePersistence
from orchestrator_agent.state import SearchState

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = get_logger(__name__)


class AGUIEventStream(_BaseAGUIEventStream[Any, Any]):
    """Custom event stream that replaces artifact-bearing tool results with their reference.

    Tool results whose metadata is a ``ToolArtifact`` are sent to the frontend as a
    lightweight JSON reference instead of the full payload, so the UI can fetch full
    results lazily.
    """

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        result = event.part
        if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
            yield ToolCallResultEvent(
                message_id=self.new_message_id(),
                type=EventType.TOOL_CALL_RESULT,
                role="tool",
                tool_call_id=result.tool_call_id,
                content=result.metadata.model_dump_json(),
            )
            return

        # Default behavior for all other tools
        async for e in super().handle_function_tool_result(event):
            yield e


class _AGUIAdapter(AGUIAdapter[Any, Any]):
    """AGUIAdapter that uses AGUIEventStream."""

    def build_event_stream(self) -> AGUIEventStream:
        return AGUIEventStream(self.run_input, accept=self.accept)


class AGUIWorker:
    """Orchestrates AG-UI request handling: state setup, persistence, stream creation."""

    @staticmethod
    async def run_request(
        agent: "WFOAgent",
        run_input: RunAgentInput,
        db_session: Session,
        *,
        auth_token: str | None = None,
    ) -> AsyncIterator[str]:
        """Execute the full AG-UI lifecycle and return an SSE event iterator."""
        prepared = AGUIWorker._prepare_run_input(run_input)

        run_id = UUID(prepared.run_id)
        thread_id = prepared.thread_id

        # Create or get agent run record
        agent_run = db_session.get(AgentRunTable, run_id)
        if not agent_run:
            agent_run = AgentRunTable(run_id=run_id, thread_id=thread_id, agent_type="search")
            db_session.add(agent_run)
            db_session.commit()
            logger.debug("Created new agent run", run_id=str(run_id), thread_id=thread_id)

        prepared.state["run_id"] = run_id

        persistence = PostgresStatePersistence(thread_id=thread_id, run_id=run_id, session=db_session)

        loaded_state = await persistence.load_state()
        if loaded_state:
            initial_state = loaded_state
            initial_state.user_input = prepared.state["user_input"]
            initial_state.run_id = run_id
            logger.debug("Loaded previous state from persistence")
        else:
            initial_state = SearchState(**prepared.state)
            logger.debug("Created fresh state (no previous snapshot)")

        adapter = _AGUIAdapter(agent=agent, run_input=prepared)

        async def _stream_with_persistence() -> AsyncIterator[str]:
            """Open the MCP session, stream AG-UI events, persist, and commit on completion."""
            try:
                with bind_outbound_token(auth_token):
                    async with agent:
                        event_stream = adapter.encode_stream(adapter.run_stream(deps=StateDeps(initial_state)))
                        async for event_str in event_stream:
                            yield event_str
                await persistence.snapshot(initial_state)
                db_session.commit()
            except Exception as e:
                logger.error("Error in agent stream", error=str(e), exc_info=True)
                db_session.rollback()
                raise

        return _stream_with_persistence()

    @staticmethod
    def _prepare_run_input(run_input: RunAgentInput) -> RunAgentInput:
        """Prepare RunAgentInput by extracting user message and adding it to state."""
        user_input = AGUIWorker._extract_user_input(run_input)

        logger.debug("Extracted latest user message", user_input=user_input[:100] if user_input else "(empty)")

        state_dict = dict(run_input.state) if run_input.state else {}
        state_dict["user_input"] = user_input

        return RunAgentInput(
            thread_id=run_input.thread_id,
            run_id=run_input.run_id,
            state=state_dict,
            messages=run_input.messages,
            tools=run_input.tools,
            context=run_input.context,
            forwarded_props=run_input.forwarded_props,
        )

    @staticmethod
    def _extract_user_input(run_input: RunAgentInput) -> str:
        """Extract the most recent user message from RunAgentInput messages."""
        for msg in reversed(run_input.messages):
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content
        return ""
