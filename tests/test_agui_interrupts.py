"""The start of a workflow over AG-UI: the waiting approval as an interrupt, and the answer a run carries on."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ag_ui.core import ActivitySnapshotEvent, ResumeEntry, RunAgentInput, UserMessage
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.ui import StateDeps

from orchestrator_agent.adapters.ag_ui import AGENT_TYPE, NOT_OPEN, AGUIEventStream, AGUIWorker
from orchestrator_agent.adapters.ag_ui.interrupts import answers, pending_interrupt, pending_start, supports_interrupts
from orchestrator_agent.adapters.ag_ui.worker import NOT_APPROVED, REJECTED, ApprovalPendingError, EmptyRunError
from orchestrator_agent.persistence import StoredThread, dump_messages
from orchestrator_agent.state import FormFillSession, SearchState

from .conftest import mock_event_stream

RUN_ID = "00000000-0000-0000-0000-000000000042"
START = ToolCallPart(tool_name="start_workflow", args={}, tool_call_id="s1")
WAITING: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content="create a port")]),
    ModelResponse(parts=[START]),
]
ANSWERED = [*WAITING, ModelRequest(parts=[ToolReturnPart(tool_name="start_workflow", content={}, tool_call_id="s1")])]


def _run_input(
    resume: list[ResumeEntry] | None = None, version: str | None = "1.0.3", messages: list[UserMessage] | None = None
) -> RunAgentInput:
    return RunAgentInput(
        thread_id="t",
        run_id=RUN_ID,
        state={},
        messages=[UserMessage(id="m1", role="user", content="create a port")] if messages is None else messages,
        tools=[],
        context=[],
        forwarded_props={},
        protocol_version=version,
        resume=resume,
    )


def _resolved(interrupt_id: str, payload: Any) -> ResumeEntry:
    return ResumeEntry(interrupt_id=interrupt_id, status="resolved", payload=payload)


def _confirming() -> SearchState:
    session = FormFillSession(workflow_key="w", status="confirming", page_inputs=[{"speed": "10G"}])
    return SearchState(form_fill=session, message_history=dump_messages(WAITING))


# --- the start that is waiting -----------------------------------------------------------------------------------


def test_the_waiting_start_is_the_last_one_without_a_result():
    assert pending_start(WAITING) is START
    assert pending_start(ANSWERED) is None and pending_start([]) is None


def test_a_reopened_chat_gets_the_waiting_start_as_the_interrupt_it_was_shown():
    interrupt = pending_interrupt(_confirming(), WAITING)
    assert interrupt is not None
    assert (interrupt.id, interrupt.reason, interrupt.tool_call_id) == ("int-s1", "tool_call", "s1")
    assert interrupt.metadata == {"workflow_key": "w", "values": {"speed": "10G"}, "defaults": {}, "summary": []}
    assert pending_interrupt(SearchState(), WAITING) is None  # no form is open
    assert pending_interrupt(_confirming(), ANSWERED) is None  # the start was decided


def test_a_run_carries_on_the_answer_to_the_waiting_start_and_no_other():
    approved = _resolved("int-s1", {"approved": True})
    stale = _resolved("int-old", {"approved": True})
    assert answers(START, [stale, approved]) == [approved]
    assert answers(None, [approved]) == []  # nothing waits: an answer is stale
    assert answers(START, None) == []  # no decision came with the run


@pytest.mark.parametrize(
    ("version", "expected"),
    [(None, False), ("", False), ("0.1.10", False), ("0.1.19", True), ("1.0.0", True), ("1.0.3", True), ("x", False)],
)
def test_supports_interrupts(version, expected):
    assert supports_interrupts(version) is expected


# --- the stream: a form as one activity message ------------------------------------------------------------------


def _stream(state: SearchState) -> AGUIEventStream:
    return AGUIEventStream(_run_input(), ag_ui_version="1.0.0", deps=StateDeps(state))


@pytest.mark.asyncio
async def test_a_form_is_one_activity_message_that_names_its_tool_calls():
    session = FormFillSession(workflow_key="w")
    state = SearchState(form_fill=session, hitl=True)
    stream = _stream(state)
    call = ToolCallPart(tool_name="submit_page", args={"a": 1}, tool_call_id="f1")
    assert [e async for e in stream.handle_tool_call_start(call)] != []  # the call itself still goes out
    result = FunctionToolResultEvent(part=ToolReturnPart(tool_name="submit_page", content={}, tool_call_id="f1"))
    moved = [e async for e in stream.handle_function_tool_result(result) if isinstance(e, ActivitySnapshotEvent)]
    assert [(e.message_id, e.content["phase"], e.content["calls"]) for e in moved] == [
        (session.id, "gathering", ["f1"])
    ]
    state.form_fill = None  # the form closed in this run
    (closed,) = [e async for e in stream.after_stream() if isinstance(e, ActivitySnapshotEvent)]
    assert (closed.message_id, closed.content["phase"], closed.content["calls"]) == (session.id, "done", ["f1"])
    # A client that cannot be shown an approval gets no form message.
    plain = _stream(SearchState(form_fill=session))
    assert [e async for e in plain.after_stream() if isinstance(e, ActivitySnapshotEvent)] == []


# --- the worker: the answer in, the run's state ------------------------------------------------------------------


def _agent() -> MagicMock:
    return MagicMock()


async def _run(mock_adapter_cls, mock_persistence_cls, state: SearchState | None, run_input: RunAgentInput):
    """One request through the worker with the adapter and the persistence mocked: what it streamed."""
    mock_persistence = AsyncMock()
    mock_persistence.load_thread.return_value = (
        StoredThread(thread_id="t", agent_type=AGENT_TYPE, updated_at=datetime.now(UTC), state=state) if state else None
    )
    mock_persistence_cls.return_value = mock_persistence

    async def no_events():
        return
        yield  # noqa: F401

    mock_adapter = MagicMock()
    mock_adapter.run_stream.return_value = mock_event_stream()
    mock_adapter.encode_stream.return_value = no_events()
    mock_adapter_cls.return_value = mock_adapter
    db_session = MagicMock()
    db_session.get.return_value = None
    stream = await AGUIWorker.run_request(_agent(), run_input, db_session)
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_the_answer_to_the_waiting_start_goes_on_and_the_run_state_is_the_servers(
    mock_adapter_cls, mock_persistence_cls
):
    approved = _resolved("int-s1", {"approved": True})
    run_input = _run_input(resume=[approved, _resolved("int-old", {"approved": True})], messages=[])
    await _run(mock_adapter_cls, mock_persistence_cls, _confirming(), run_input)
    prepared = mock_adapter_cls.call_args.kwargs["run_input"]
    assert prepared.resume == [approved] and prepared.messages == []
    assert prepared.state["form_fill"]["workflow_key"] == "w" and prepared.state["hitl"] is True
    assert mock_adapter_cls.call_args.kwargs["deps"].state.hitl is True


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_message_without_a_decision_while_the_start_waits_is_refused(mock_adapter_cls, mock_persistence_cls):
    said = [UserMessage(id="m2", role="user", content="show me subscriptions over time")]
    with pytest.raises(ApprovalPendingError, match="int-s1"):
        await _run(mock_adapter_cls, mock_persistence_cls, _confirming(), _run_input(messages=said))
    assert not mock_adapter_cls.called


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_what_the_person_wrote_when_not_approving_reaches_the_model_as_theirs(
    mock_adapter_cls, mock_persistence_cls
):
    state = _confirming()
    talk = _resolved("int-s1", {"approved": False, "reason": "show me subscriptions over time"})
    await _run(mock_adapter_cls, mock_persistence_cls, state, _run_input(resume=[talk], messages=[]))
    (told,) = mock_adapter_cls.call_args.kwargs["run_input"].resume
    assert told.payload["reason"] == f"{NOT_APPROVED} 'show me subscriptions over time'"
    assert state.form_fill is not None  # the form stays open: the start can be asked for again


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_rejection_without_words_closes_the_form_and_the_model_is_told(mock_adapter_cls, mock_persistence_cls):
    state = _confirming()
    form = (state.form_fill.id, "w")
    rejected = _run_input(resume=[_resolved("int-s1", {"approved": False})], messages=[])
    await _run(mock_adapter_cls, mock_persistence_cls, state, rejected)
    kwargs = mock_adapter_cls.call_args.kwargs
    assert kwargs["deps"].state.form_fill is None and kwargs["closed"] == form
    assert kwargs["run_input"].resume[0].payload == {"approved": False, "reason": REJECTED}


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_an_answer_to_a_start_that_is_not_waiting_runs_nothing(mock_adapter_cls, mock_persistence_cls):
    stale = _run_input(resume=[_resolved("int-old", {"approved": True})], messages=[])
    chunks = await _run(mock_adapter_cls, mock_persistence_cls, SearchState(), stale)
    assert NOT_OPEN in "".join(chunks) and not mock_adapter_cls.called


@pytest.mark.asyncio
async def test_a_request_with_neither_a_message_nor_an_answer_is_refused():
    with pytest.raises(EmptyRunError):
        await AGUIWorker.run_request(_agent(), _run_input(messages=[]), MagicMock())


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_client_without_interrupts_cannot_be_shown_an_approval(mock_adapter_cls, mock_persistence_cls):
    await _run(mock_adapter_cls, mock_persistence_cls, SearchState(), _run_input(version="0.1.10"))
    assert mock_adapter_cls.call_args.kwargs["deps"].state.hitl is False
