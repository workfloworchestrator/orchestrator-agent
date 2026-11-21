"""The start of a workflow over A2A: approved through kagent's HITL extension."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueueLegacy
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState, TaskStatusUpdateEvent
from google.protobuf import json_format
from pydantic_ai import ToolDenied
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

from orchestrator_agent.adapters.a2a import WFOAgentExecutor
from orchestrator_agent.adapters.a2a.kagent import HITL_EXTENSION_URI
from orchestrator_agent.form_fill.start import NOT_APPROVED, REJECTED
from orchestrator_agent.persistence import dump_messages
from orchestrator_agent.state import FormFillSession, FormPhase, SearchState

from .conftest import make_text_result_event, mock_event_stream

START = ToolCallPart(tool_name="start_workflow", args={}, tool_call_id="s1")
WAITING: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content="create a port")]),
    ModelResponse(parts=[START]),
]


def _waiting() -> SearchState:
    """A thread whose form is complete and whose start waits for the person."""
    session = FormFillSession(workflow_key="w", status=FormPhase.CONFIRMING, page_inputs=[{"speed": "10G"}])
    return SearchState(form_fill=session, message_history=dump_messages(WAITING))


def _context(text: str = "create a port", *, response: dict[str, Any] | None = None, extension: bool = True):
    message = Message(role=Role.ROLE_USER, parts=[Part(text=text)], message_id=str(uuid.uuid4()))
    if response is not None:
        message.metadata.update({HITL_EXTENSION_URI: {"type": "tool_approval_response", "approvals": [response]}})
    call = ServerCallContext(requested_extensions={HITL_EXTENSION_URI} if extension else set())
    return RequestContext(call_context=call, request=SendMessageRequest(message=message))


async def _execute(context: RequestContext, prior: SearchState | None, setup=lambda state: None):
    """One call of the executor: the agent mock (what it was run with) and the task's last status."""
    agent = MagicMock()

    @asynccontextmanager
    async def _run(*args, **kwargs):
        agent._last_call = (args, kwargs)
        setup(kwargs["deps"].state)
        yield mock_event_stream(make_text_result_event(""))

    agent.run_stream_events = MagicMock(side_effect=_run)
    queue = EventQueueLegacy()
    with (
        patch("orchestrator_agent.adapters.a2a.adapter.PostgresStatePersistence") as persistence,
        patch("orchestrator.core.db.db"),
    ):
        persistence.return_value.load_state = AsyncMock(return_value=prior)
        persistence.return_value.snapshot = AsyncMock()
        await WFOAgentExecutor(agent).execute(context, queue)
    statuses = []
    while not queue.queue.empty():
        event = queue.queue.get_nowait()
        if isinstance(event, TaskStatusUpdateEvent):
            statuses.append(event.status)
    return agent, statuses[-1]


@pytest.mark.asyncio
async def test_a_run_that_stops_on_the_start_pauses_the_task_with_the_start_to_approve():
    def stops(state: SearchState) -> None:
        state.form_fill = _waiting().form_fill

    with patch("orchestrator_agent.turn.pending_start", return_value=START):
        agent, status = await _execute(_context(), None, stops)
    assert status.state == TaskState.TASK_STATE_INPUT_REQUIRED and agent._last_call[1]["deps"].state.hitl
    assert list(status.message.extensions) == [HITL_EXTENSION_URI]  # kagent reads the payload only with it
    payload = json_format.MessageToDict(status.message.metadata)[HITL_EXTENSION_URI]
    assert payload["type"] == "tool_approval_request" and payload["hint"] == status.message.parts[0].text
    assert payload["tools"] == [
        {
            "id": "s1",
            "call_id": "s1",
            "name": "start_workflow",
            "args": {"workflow_key": "w", "values": {"speed": "10G"}},
        }
    ]


@pytest.mark.parametrize(
    "response, result, closed",
    [
        pytest.param({"id": "s1", "approved": True}, True, False, id="approve"),
        pytest.param({"id": "s1", "approved": False}, ToolDenied(REJECTED), True, id="reject"),
        pytest.param(
            {"id": "s1", "approved": False, "rejection_reason": "make it 1G"},
            ToolDenied(f"{NOT_APPROVED} 'make it 1G'"),
            False,
            id="reject-with-a-reason",
        ),
    ],
)
@pytest.mark.asyncio
async def test_the_response_is_the_result_of_the_waiting_start(response, result, closed):
    agent, status = await _execute(_context("Human input supplied", response=response), _waiting())
    args, kwargs = agent._last_call
    assert args == (None,) and kwargs["deferred_tool_results"].approvals == {"s1": result}
    assert (kwargs["deps"].state.form_fill is None) is closed and status.state == TaskState.TASK_STATE_COMPLETED


@pytest.mark.asyncio
async def test_a_message_without_a_decision_shows_the_waiting_start_again():
    # kagent shows one pause per call: a start asked for again on a resumed call is shown on the next call.
    agent, status = await _execute(_context("continue"), _waiting())
    assert status.state == TaskState.TASK_STATE_INPUT_REQUIRED and not agent.run_stream_events.called


@pytest.mark.asyncio
async def test_a_caller_without_the_extension_cannot_approve_and_its_message_declines_a_waiting_start():
    agent, _ = await _execute(_context("never mind", extension=False), _waiting())
    args, kwargs = agent._last_call
    assert kwargs["deps"].state.hitl is False
    assert kwargs["deferred_tool_results"].approvals == {"s1": ToolDenied(f"{NOT_APPROVED} 'never mind'")}
