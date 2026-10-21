"""Tests for the AG-UI chats: threads bound to their owner, history replayed and recorded, and the chats API."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ag_ui.core import ResumeEntry, RunAgentInput, UserMessage
from fastapi import FastAPI
from fastapi.testclient import TestClient
from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.ui import StateDeps

from orchestrator_agent.adapters.ag_ui import (
    AGENT_TYPE,
    AGUIEventStream,
    AGUIInterrupts,
    AGUIWorker,
    ThreadForbiddenError,
)
from orchestrator_agent.adapters.ag_ui.chats import UNTITLED, pending_stop, title_of, transcript
from orchestrator_agent.adapters.ag_ui.interrupts import pause
from orchestrator_agent.adapters.words import shown
from orchestrator_agent.api.endpoints import chats
from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.form_fill.widgets import Option, Widget
from orchestrator_agent.persistence import StoredThread, dump_messages
from orchestrator_agent.state import AskField, FormFillSession, Reply, SearchState

from .conftest import mock_event_stream

RUN_ID = "00000000-0000-0000-0000-000000000042"
GATHERING = Reply(
    '{"workflow_key":"w","status":"gathering","title":"Page"}',
    ask=[AskField(name="redundancy", question="Redundancy?", choices=["Protected"], values=["protected"])],
)
ARTIFACT = QueryArtifact(description="search returned 2 result(s)", query_id="q1", total_results=2)


def _history() -> list[dict[str, Any]]:
    """A thread's memory: a question, a search with an artifact, an answer, and a form-fill reply."""
    return dump_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="  find  two subscriptions ")]),
            ModelResponse(parts=[ToolCallPart(tool_name="search", args={"q": "x"}, tool_call_id="c1")]),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="search", content={"rows": [1, 2]}, tool_call_id="c1", metadata=ARTIFACT)
                ]
            ),
            ModelResponse(parts=[TextPart(content="Two found.")]),
            ModelRequest(parts=[UserPromptPart(content="start a workflow")]),
            ModelResponse(parts=[TextPart(content=GATHERING.text)]),
        ]
    )


# --- the chat as the popup reads it back --------------------------------------------------------------------


def test_the_title_is_the_first_question_cut_short():
    assert title_of(SearchState(message_history=_history())) == "find two subscriptions"
    assert title_of(SearchState()) == UNTITLED
    long = dump_messages([ModelRequest(parts=[UserPromptPart(content="x" * 200)])])
    assert len(title_of(SearchState(message_history=long))) == 80


def test_the_transcript_is_agui_messages_with_artifact_references_and_the_persons_words():
    messages = transcript(SearchState(message_history=_history()))
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert messages[2].content == ARTIFACT.model_dump_json()  # the reference the stream sent, not the raw rows
    assert messages[3].content == "Two found."
    assert messages[5].content == shown(GATHERING)  # the skill's JSON, as the person read it


def test_a_tools_note_to_the_model_is_not_in_the_transcript_nor_the_title():
    # pydantic-ai files ToolReturn.content ("a chart is shown below your reply...") as a user prompt part
    # next to the tool return; nobody typed it, so the popup never shows it.
    note = "A chart for this result is shown to the user automatically below your reply."
    history = dump_messages(
        [
            ModelResponse(parts=[ToolCallPart(tool_name="aggregate", args={}, tool_call_id="c1")]),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="aggregate", content={"rows": []}, tool_call_id="c1", metadata=ARTIFACT),
                    UserPromptPart(content=note),
                ]
            ),
            ModelResponse(parts=[TextPart(content="A steady rise.")]),
        ]
    )
    state = SearchState(message_history=history)
    assert [m.role for m in transcript(state)] == ["assistant", "tool", "assistant"]
    assert title_of(state) == UNTITLED


def test_the_transcript_can_be_its_last_messages_only():
    messages = transcript(SearchState(message_history=_history()), last=2)
    assert [m.role for m in messages] == ["user", "assistant"]


def test_the_pending_stop_is_the_interrupt_as_shown_while_the_form_is_open():
    session = FormFillSession(workflow_key="w")
    interrupt = pause(GATHERING, session)
    assert interrupt is not None
    state = SearchState(form_fill=session)
    assert pending_stop(state) == interrupt.model_dump(mode="json", by_alias=True, exclude_none=True)
    session.status = "done"
    assert pending_stop(state) is None
    assert pending_stop(SearchState()) is None


# --- the worker: owner binding, history in and out -----------------------------------------------------------


def _run_input(resume: list[ResumeEntry] | None = None) -> RunAgentInput:
    return RunAgentInput(
        thread_id="t",
        run_id=RUN_ID,
        state={},
        messages=[UserMessage(id="m1", role="user", content="hello")],
        tools=[],
        context=[],
        forwarded_props={},
        protocol_version="1.0.3",
        resume=resume,
    )


def _stored(state: SearchState, agent_type: str = AGENT_TYPE) -> StoredThread:
    return StoredThread(thread_id="t", agent_type=agent_type, updated_at=datetime.now(UTC), state=state)


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    return agent


def _persistence(mock_persistence_cls: MagicMock, thread: StoredThread | None) -> AsyncMock:
    mock_persistence = AsyncMock()
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


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_new_thread_is_the_callers_and_recorded_as_this_routes(mock_adapter_cls, mock_persistence_cls):
    _persistence(mock_persistence_cls, None)
    _adapter(mock_adapter_cls)
    db_session = MagicMock()
    db_session.get.return_value = None

    stream = await AGUIWorker.run_request(_agent(), _run_input(), db_session, owner="alice")
    _ = [chunk async for chunk in stream]

    assert mock_adapter_cls.call_args.kwargs["deps"].state.owner == "alice"
    assert db_session.add.call_args.args[0].agent_type == AGENT_TYPE


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_the_threads_history_is_replayed_and_the_runs_messages_recorded(mock_adapter_cls, mock_persistence_cls):
    mock_persistence = _persistence(
        mock_persistence_cls, _stored(SearchState(owner="alice", message_history=_history()))
    )
    mock_adapter = _adapter(mock_adapter_cls)
    db_session = MagicMock()
    db_session.get.return_value = None

    stream = await AGUIWorker.run_request(_agent(), _run_input(), db_session, owner="alice")
    _ = [chunk async for chunk in stream]

    replayed = mock_adapter.run_stream.call_args.kwargs["message_history"]
    assert len(replayed) == 6 and isinstance(replayed[0], ModelRequest)
    mock_persistence.snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_run_result_becomes_the_threads_memory():
    state = SearchState()
    stream = AGUIEventStream(_run_input(), ag_ui_version="1.0.0", hitl=AGUIInterrupts(), deps=StateDeps(state))
    result = MagicMock()
    result.all_messages.return_value = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    _ = [event async for event in stream.handle_run_result(AgentRunResultEvent(result=result))]
    assert state.message_history == dump_messages(result.all_messages.return_value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread", "owner"),
    [
        pytest.param(_stored(SearchState(owner="alice")), "bob", id="someone-elses"),
        pytest.param(_stored(SearchState(owner="alice")), None, id="owned-but-anonymous-caller"),
        pytest.param(_stored(SearchState(), agent_type="a2a"), "alice", id="another-routes"),
    ],
)
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_a_thread_that_is_not_the_callers_is_refused(mock_adapter_cls, mock_persistence_cls, thread, owner):
    _persistence(mock_persistence_cls, thread)
    _adapter(mock_adapter_cls)
    db_session = MagicMock()
    with pytest.raises(ThreadForbiddenError):
        await AGUIWorker.run_request(_agent(), _run_input(), db_session, owner=owner)
    mock_adapter_cls.assert_not_called()
    db_session.add.assert_not_called()


@pytest.mark.asyncio
@patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence")
@patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter")
async def test_an_unowned_thread_of_this_route_becomes_the_callers(mock_adapter_cls, mock_persistence_cls):
    _persistence(mock_persistence_cls, _stored(SearchState(), agent_type="search"))
    _adapter(mock_adapter_cls)
    db_session = MagicMock()
    db_session.get.return_value = None

    stream = await AGUIWorker.run_request(_agent(), _run_input(), db_session, owner="alice")
    _ = [chunk async for chunk in stream]

    assert mock_adapter_cls.call_args.kwargs["deps"].state.owner == "alice"


# --- the API -------------------------------------------------------------------------------------------------------


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chats.router, prefix="/chats")
    with patch("orchestrator_agent.api.endpoints.chats.db") as mock_db:
        mock_db.session = MagicMock()
        yield TestClient(app)


def _threads() -> list[StoredThread]:
    session = FormFillSession(workflow_key="w")
    pause(GATHERING, session)
    return [
        StoredThread(
            "t1", AGENT_TYPE, datetime(2026, 10, 20, 12, 0, tzinfo=UTC), SearchState(message_history=_history())
        ),
        StoredThread("t2", AGENT_TYPE, datetime(2026, 10, 20, 11, 0, tzinfo=UTC), SearchState(form_fill=session)),
    ]


def test_the_list_is_the_callers_chats_of_this_route(client):
    with patch("orchestrator_agent.api.endpoints.chats.list_threads", return_value=_threads()) as listed:
        response = client.get("/chats/?limit=10")
    assert response.status_code == 200
    assert [(c["thread_id"], c["title"], c["form_open"]) for c in response.json()] == [
        ("t1", "find two subscriptions", False),
        ("t2", UNTITLED, True),
    ]
    assert listed.call_args.kwargs["agent_types"] == ("agui", "search")
    assert listed.call_args.kwargs["owner"] is None  # no OAuth2: the unowned chats
    assert listed.call_args.kwargs["limit"] == 10


def test_a_chat_is_its_last_messages_and_its_pending_stop(client):
    thread = _threads()[1]
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=thread):
        response = client.get("/chats/t2?last=5")
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "t2" and body["form_open"] is True
    assert body["pending"] == pending_stop(thread.state)
    assert body["messages"] == []


def test_a_chats_messages_come_as_agui_messages(client):
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=_threads()[0]):
        response = client.get("/chats/t1?last=2")
    messages = response.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == shown(GATHERING)


@pytest.mark.parametrize(
    ("thread", "status"),
    [
        pytest.param(None, 404, id="unknown"),
        pytest.param(StoredThread("t", "a2a", datetime.now(UTC), SearchState()), 404, id="another-routes"),
        pytest.param(
            StoredThread("t", AGENT_TYPE, datetime.now(UTC), SearchState(owner="alice")), 403, id="someone-elses"
        ),
    ],
)
def test_a_chat_that_is_not_the_callers_is_not_served(client, thread, status):
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=thread):
        assert client.get("/chats/t").status_code == status
        assert client.delete("/chats/t").status_code == status


def test_forgetting_a_chat_deletes_it(client):
    thread = _threads()[0]
    with (
        patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=thread),
        patch("orchestrator_agent.api.endpoints.chats.delete_thread", return_value=2) as deleted,
    ):
        response = client.delete("/chats/t1")
    assert response.status_code == 204
    assert deleted.call_args.args[1] == "t1"


# --- a searchable question's options -------------------------------------------------------------------------


class _CustomerWidget(Widget):
    """A stand-in for the customer widget: three customers, narrowed like the real one."""

    id = "customerId"

    def matches(self, field):
        return field.get("format") == "customerId"

    async def fetch(self, field, ctx):
        return [
            Option("c-1", "Alder College (UA)"),
            Option("c-2", "Birch University (UB)"),
            Option("c-3", "Cedar Labs"),
        ]


def _searching_thread() -> StoredThread:
    """A thread paused on a page whose customer field is a long list the person may search."""
    page = WorkflowFormPage(
        page=0,
        complete=False,
        status="next",
        schema={"properties": {"customer_id": {"type": "string", "format": "customerId", "title": "Customer"}}},
    )
    session = FormFillSession(workflow_key="w", pages=[page])
    asked = Reply(GATHERING.text, ask=[AskField(name="customer_id", question="Customer *", searchable=True)])
    pause(asked, session)
    return StoredThread("t3", AGENT_TYPE, datetime(2026, 10, 20, 11, 0, tzinfo=UTC), SearchState(form_fill=session))


@pytest.fixture
def searching_client(client):
    client.app.state.form_fill = FormFillSkill(widgets=[_CustomerWidget()], graphql=AsyncMock())
    client.app.state.core_toolset = SimpleNamespace(direct_call_tool=AsyncMock())
    return client


def test_the_options_a_search_fits_come_from_the_fields_widget(searching_client):
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=_searching_thread()):
        response = searching_client.get("/chats/t3/options/customer_id?q=ua")
    assert response.status_code == 200
    assert response.json() == [{"value": "c-1", "label": "Alder College (UA)"}]


def test_without_words_the_first_options_are_offered(searching_client):
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=_searching_thread()):
        response = searching_client.get("/chats/t3/options/customer_id?limit=2")
    assert [option["value"] for option in response.json()] == ["c-1", "c-2"]


def test_the_search_runs_as_the_person_asking(searching_client):
    with (
        patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=_searching_thread()),
        patch("orchestrator_agent.api.endpoints.chats.bind_outbound_token") as bound,
    ):
        searching_client.get("/chats/t3/options/customer_id?q=cedar", headers={"Authorization": "Bearer tok-1"})
    assert bound.call_args.args == ("tok-1",)


def test_only_a_searchable_question_of_the_pending_stop_is_searched(searching_client):
    thread = _searching_thread()
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=thread):
        assert searching_client.get("/chats/t3/options/port?q=a").status_code == 404
    with patch("orchestrator_agent.api.endpoints.chats.load_thread", return_value=_threads()[0]):  # no form open
        assert searching_client.get("/chats/t1/options/customer_id?q=a").status_code == 404
