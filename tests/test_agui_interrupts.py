"""Tests for the AG-UI interrupt transport: the approval out as an interrupt, the resume entry in as the skill's input."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ag_ui.core import ResumeEntry, RunAgentInput, RunFinishedEvent, TextMessageContentEvent, UserMessage
from pydantic_ai.messages import TextPart
from pydantic_ai.ui import StateDeps

from orchestrator_agent.adapters.ag_ui import AGENT_TYPE, NOT_OPEN, AGUIEventStream, AGUIInterrupts, AGUIWorker
from orchestrator_agent.adapters.ag_ui.interrupts import APPROVAL_REASON, human_input, pause, supports_interrupts
from orchestrator_agent.adapters.words import shown
from orchestrator_agent.form_fill.pending import PendingStop, pending_approval
from orchestrator_agent.persistence import StoredThread
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply, SearchState

from .conftest import mock_event_stream

APPROVAL = Approval(hint="Start it?", tool_name="create_workflow", args={"workflow_key": "w", "json_data": [{"a": 1}]})
CONFIRMING = Reply('{"workflow_key":"w","status":"confirming","values":{"a":1}}', approval=APPROVAL)
ENDED = Reply('{"workflow_key":"w","status":"started","process_id":"p1"}')
RUN_ID = "00000000-0000-0000-0000-000000000042"


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


def _paused() -> tuple[FormFillSession, str]:
    """A session paused at the approval, and the interrupt's id."""
    session = FormFillSession(workflow_key="w", status="confirming")
    interrupt = pause(CONFIRMING, session)
    assert interrupt is not None
    return session, interrupt.id


def _resolved(interrupt_id: str, payload: Any) -> ResumeEntry:
    return ResumeEntry(interrupt_id=interrupt_id, status="resolved", payload=payload)


# --- outbound ----------------------------------------------------------------------------------------------------


def test_the_start_is_a_tool_call_interrupt_shown_as_it_will_be_made():
    session = FormFillSession(workflow_key="w", status="confirming")
    interrupt = pause(CONFIRMING, session)
    assert interrupt is not None
    assert interrupt.reason == APPROVAL_REASON
    assert interrupt.tool_call_id == interrupt.id
    assert interrupt.message == "Start it?"
    assert (interrupt.metadata or {})["tool"] == {"name": "create_workflow", "args": APPROVAL.args}
    assert (interrupt.metadata or {})["form"] == json.loads(CONFIRMING.text)
    assert interrupt.response_schema is not None and set(interrupt.response_schema["properties"]) == {
        "approved",
        "reason",
    }
    # The stop is remembered as it was shown (for a reopened chat), and is seen at once.
    remembered = PendingStop.model_validate(session.pending)
    assert remembered.model_dump(exclude={"shown"}) == pending_approval(interrupt.id).model_dump(exclude={"shown"})
    assert remembered.shown is not None and remembered.shown["toolCallId"] == interrupt.id
    assert session.unseen is False


def test_a_reply_that_is_no_stop_pauses_nothing():
    session = FormFillSession(workflow_key="w")
    assert pause(ENDED, session) is None
    assert session.pending is None
    assert pause(CONFIRMING, None) is None


# --- inbound -----------------------------------------------------------------------------------------------------


def test_an_approval_is_a_start():
    session, interrupt_id = _paused()
    assert human_input(session, [_resolved(interrupt_id, {"approved": True})]) == FormInput(decision=Decision.START)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        pytest.param(None, None, id="no-words"),
        pytest.param("  ", None, id="blank-words"),
        pytest.param("make it 1G", "make it 1G", id="what-to-change"),
    ],
)
def test_a_rejection_carries_what_the_person_said(reason, expected):
    session, interrupt_id = _paused()
    entry = _resolved(interrupt_id, {"approved": False, "reason": reason})
    assert human_input(session, [entry]) == FormInput(decision=Decision.CANCEL, reason=expected)


def test_a_cancelled_entry_rejects_the_start():
    session, interrupt_id = _paused()
    entry = ResumeEntry(interrupt_id=interrupt_id, status="cancelled")
    assert human_input(session, [entry]) == FormInput(decision=Decision.CANCEL)


def test_an_entry_about_another_interrupt_shows_the_stop_again():
    session, _ = _paused()
    assert human_input(session, [_resolved("someone-elses", {"approved": True})]) == FormInput()


@pytest.mark.parametrize(
    "payload",
    [pytest.param({"approved": "yes"}, id="approval-not-a-boolean"), pytest.param(None, id="no-payload")],
)
def test_an_unreadable_response_shows_the_stop_again(payload):
    session, interrupt_id = _paused()
    assert human_input(session, [_resolved(interrupt_id, payload)]) == FormInput()


def test_no_entries_or_no_pending_stop_is_no_response():
    session, interrupt_id = _paused()
    assert human_input(session, None) is None
    assert human_input(session, []) is None
    assert human_input(FormFillSession(workflow_key="w"), [_resolved(interrupt_id, {"approved": True})]) is None
    assert human_input(None, [_resolved(interrupt_id, {"approved": True})]) is None


@pytest.mark.parametrize(
    ("version", "expected"),
    [(None, False), ("", False), ("0.1.10", False), ("0.1.19", True), ("1.0.0", True), ("1.0.3", True), ("x", False)],
)
def test_supports_interrupts(version, expected):
    assert supports_interrupts(version) is expected


def test_the_transport_claims_a_client_whose_protocol_version_has_interrupts():
    transport = AGUIInterrupts()
    assert transport.shows_stops(_run_input(version="1.0.3")) is True
    assert transport.shows_stops(_run_input(version="0.1.10")) is False
    assert transport.shows_stops(_run_input(version=None)) is False
    assert transport.words(CONFIRMING) == shown(CONFIRMING)


# --- the event stream: the run's end and the reply's text ----------------------------------------------------


TRANSPORT = AGUIInterrupts()


def _stream(state: SearchState, hitl: AGUIInterrupts | None = TRANSPORT) -> AGUIEventStream:
    return AGUIEventStream(_run_input(), ag_ui_version="1.0.0", hitl=hitl, deps=StateDeps(state))


@pytest.mark.asyncio
async def test_a_stop_ends_the_run_with_an_interrupt_outcome_recorded_on_the_session():
    state = SearchState(form_fill=FormFillSession(workflow_key="w", status="confirming"), form_reply=CONFIRMING)
    events = [event async for event in _stream(state).after_stream()]
    # A skipped model request streams no text, so the reply is said here, in the person's words, before the stop.
    deltas = [event.delta for event in events if isinstance(event, TextMessageContentEvent)]
    assert deltas == [shown(CONFIRMING)]
    assert isinstance(events[-1], RunFinishedEvent)
    outcome = events[-1].outcome
    assert outcome is not None and outcome.type == "interrupt"
    (interrupt,) = outcome.interrupts
    assert interrupt.reason == APPROVAL_REASON
    assert state.form_fill is not None and state.form_fill.pending["id"] == interrupt.id


@pytest.mark.asyncio
async def test_a_reply_that_ends_the_form_finishes_the_run_plainly():
    state = SearchState(form_fill=FormFillSession(workflow_key="w"), form_reply=ENDED)
    events = [event async for event in _stream(state).after_stream()]
    deltas = [event.delta for event in events if isinstance(event, TextMessageContentEvent)]
    assert deltas == [shown(ENDED)]
    assert isinstance(events[-1], RunFinishedEvent) and events[-1].outcome is not None
    assert events[-1].outcome.type == "success"


@pytest.mark.asyncio
async def test_without_a_transport_a_stop_finishes_the_run_plainly_and_records_nothing():
    state = SearchState(form_fill=FormFillSession(workflow_key="w", status="confirming"), form_reply=CONFIRMING)
    events = [event async for event in _stream(state, hitl=None).after_stream()]
    assert isinstance(events[-1], RunFinishedEvent) and events[-1].outcome is not None
    assert events[-1].outcome.type == "success"
    assert state.form_fill is not None and state.form_fill.pending is None


@pytest.mark.asyncio
async def test_the_skills_reply_is_streamed_in_the_transports_words():
    state = SearchState(form_reply=CONFIRMING)
    events = [event async for event in _stream(state).handle_text_start(TextPart(content=CONFIRMING.text))]
    deltas = [event.delta for event in events if isinstance(event, TextMessageContentEvent)]
    assert deltas == [shown(CONFIRMING)]


@pytest.mark.asyncio
async def test_the_models_text_is_streamed_as_it_is():
    events = [event async for event in _stream(SearchState()).handle_text_start(TextPart(content="hello"))]
    deltas = [event.delta for event in events if isinstance(event, TextMessageContentEvent)]
    assert deltas == ["hello"]


@pytest.mark.asyncio
async def test_a_reply_already_streamed_as_text_is_not_said_twice():
    state = SearchState(form_fill=FormFillSession(workflow_key="w"), form_reply=ENDED)
    stream = _stream(state)
    _ = [event async for event in stream.handle_text_start(TextPart(content=ENDED.text))]
    events = [event async for event in stream.after_stream()]
    assert not any(isinstance(event, TextMessageContentEvent) for event in events)


# --- the worker: the response in, the run's state ----------------------------------------------------------------


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    return agent


def _persistence(mock_persistence_cls: MagicMock, state: SearchState | None) -> AsyncMock:
    mock_persistence = AsyncMock()
    thread = (
        StoredThread(thread_id="t", agent_type=AGENT_TYPE, updated_at=datetime.now(UTC), state=state) if state else None
    )
    mock_persistence.load_thread.return_value = thread
    mock_persistence_cls.return_value = mock_persistence
    return mock_persistence


def _adapter(mock_adapter_cls: MagicMock) -> MagicMock:
    async def no_events():
        return
        yield  # noqa: F401

    mock_adapter = MagicMock()
    mock_adapter.run_stream.return_value = mock_event_stream()
    mock_adapter.encode_stream.return_value = no_events()
    mock_adapter_cls.return_value = mock_adapter
    return mock_adapter


async def _run(mock_adapter_cls, mock_persistence_cls, state: SearchState | None, run_input: RunAgentInput):
    mock_persistence = _persistence(mock_persistence_cls, state)
    _adapter(mock_adapter_cls)
    db_session = MagicMock()
    db_session.get.return_value = None
    stream = await AGUIWorker.run_request(_agent(), run_input, db_session)
    chunks = [chunk async for chunk in stream]
    return mock_persistence, chunks


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_the_resume_entry_is_the_skills_input_and_the_run_state_is_the_servers(
    mock_adapter_cls, mock_persistence_cls
):
    session, interrupt_id = _paused()
    run_input = _run_input(resume=[_resolved(interrupt_id, {"approved": True})], messages=[])
    mock_persistence, _ = await _run(mock_adapter_cls, mock_persistence_cls, SearchState(form_fill=session), run_input)

    kwargs = mock_adapter_cls.call_args.kwargs
    deps = kwargs["deps"]
    assert deps.state.hitl is True
    assert deps.state.form_input == FormInput(decision=Decision.START)
    assert deps.state.user_input == "[form response]"
    assert kwargs["ag_ui_version"] == "1.0.3"
    assert isinstance(kwargs["hitl"], AGUIInterrupts)
    # pydantic-ai rebuilds the run's state from the run input: it carries the server's state, this turn's input included.
    prepared = kwargs["run_input"]
    assert prepared.state["form_fill"]["pending"]["id"] == interrupt_id
    assert prepared.resume is None  # read into the state; pydantic-ai maps entries to its own interrupts only
    assert prepared.state["hitl"] is True and prepared.state["form_input"] == {"decision": "start", "reason": None}
    assert prepared.messages == []  # an approval needs no words from the person
    mock_persistence.snapshot.assert_awaited_once_with(deps.state)


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_what_the_person_said_when_rejecting_is_the_runs_message(mock_adapter_cls, mock_persistence_cls):
    session, interrupt_id = _paused()
    entry = _resolved(interrupt_id, {"approved": False, "reason": "make it 1G"})
    await _run(
        mock_adapter_cls, mock_persistence_cls, SearchState(form_fill=session), _run_input(resume=[entry], messages=[])
    )
    prepared = mock_adapter_cls.call_args.kwargs["run_input"]
    assert [(m.role, m.content) for m in prepared.messages] == [("user", "make it 1G")]
    assert mock_adapter_cls.call_args.kwargs["deps"].state.form_input == FormInput(
        decision=Decision.CANCEL, reason="make it 1G"
    )


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_response_to_a_form_that_is_not_open_runs_nothing(mock_adapter_cls, mock_persistence_cls):
    run_input = _run_input(resume=[_resolved("gone", {"approved": True})], messages=[])
    _, chunks = await _run(mock_adapter_cls, mock_persistence_cls, SearchState(), run_input)
    mock_adapter_cls.assert_not_called()
    assert any("RUN_STARTED" in chunk for chunk in chunks)
    assert any(NOT_OPEN in chunk for chunk in chunks)
    assert any("RUN_FINISHED" in chunk for chunk in chunks)


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_message_with_a_stale_response_is_still_the_models(mock_adapter_cls, mock_persistence_cls):
    run_input = _run_input(resume=[_resolved("gone", {"approved": False})])  # and "create a port"
    await _run(mock_adapter_cls, mock_persistence_cls, SearchState(), run_input)
    assert mock_adapter_cls.call_args.kwargs["deps"].state.user_input == "create a port"


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_client_without_interrupts_cannot_be_shown_a_stop(mock_adapter_cls, mock_persistence_cls):
    await _run(mock_adapter_cls, mock_persistence_cls, None, _run_input(version="0.1.10"))
    kwargs = mock_adapter_cls.call_args.kwargs
    assert kwargs["hitl"] is None
    assert kwargs["deps"].state.hitl is False
    assert kwargs["deps"].state.user_input == "create a port"
