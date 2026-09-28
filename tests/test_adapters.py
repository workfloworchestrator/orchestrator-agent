"""Adapter output tests — verify each protocol adapter transforms agent events correctly.

We mock `agent.run_stream_events()` to yield pre-built event sequences.
No LLM calls, no DB calls — just adapter transformation logic.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from ag_ui.core import RunAgentInput, ToolCallResultEvent, UserMessage
from google.protobuf import json_format
from pydantic_ai.messages import ToolReturnPart

from orchestrator_agent.adapters.a2a import A2A_SKILLS, NO_RESULTS, A2AAdapter, WFOAgentExecutor
from orchestrator_agent.adapters.ag_ui import AGUIEventStream, AGUIWorker, _AGUIAdapter
from orchestrator_agent.adapters.kagent_hitl import HITL_EXTENSION_URI
from orchestrator_agent.adapters.mcp import MCPApp, MCPWorker
from orchestrator_agent.adapters.stream import collect_stream_output
from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.state import FormFillSession, Reply, SearchState

from .conftest import (
    make_artifact_event,
    make_non_artifact_event,
    make_text_result_event,
    minimal_run_input,
    mock_event_stream,
)

SAMPLE_ARTIFACT = QueryArtifact(
    description="Found 5 subscriptions",
    query_id="q-123",
    total_results=5,
)


def _agent_doing(setup, text="model text") -> MagicMock:
    """An agent mock whose run applies ``setup(state)`` (what the form-fill capability would do) and answers ``text``."""
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)

    @asynccontextmanager
    async def _run(*args, **kwargs):
        agent._last_call = (args, kwargs)
        setup(kwargs["deps"].state)
        yield mock_event_stream(make_text_result_event(text))

    agent.run_stream_events = MagicMock(side_effect=_run)
    return agent


def _agent_mock(event_stream_factory) -> MagicMock:
    """Build an agent mock that is an async context manager and streams the given events.

    `run_stream_events(...)` must return an async context manager (matching pydantic-ai),
    so we wrap the event iterator in an async CM.
    """
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)

    @asynccontextmanager
    async def _run_stream_events(*args, **kwargs):
        agent._last_call = (args, kwargs)
        yield event_stream_factory()

    agent.run_stream_events = MagicMock(side_effect=_run_stream_events)
    return agent


class TestCollectStreamOutput:
    @pytest.mark.asyncio
    async def test_artifacts_take_priority_over_text(self):
        stream = mock_event_stream(
            make_artifact_event("search", SAMPLE_ARTIFACT),
            make_text_result_event("Execution completed"),
        )
        result = await collect_stream_output(stream)
        assert result == SAMPLE_ARTIFACT.model_dump_json()

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        result = await collect_stream_output(mock_event_stream())
        assert result == "No results"


class TestAGUIEventStream:
    def _make_stream(self) -> AGUIEventStream:
        return AGUIEventStream(run_input=minimal_run_input())

    @pytest.mark.asyncio
    async def test_artifact_becomes_lightweight_json(self):
        stream = self._make_stream()
        event = make_artifact_event("search", SAMPLE_ARTIFACT)

        results = [e async for e in stream.handle_function_tool_result(event)]

        assert isinstance(results[0], ToolCallResultEvent)
        assert results[0].content == SAMPLE_ARTIFACT.model_dump_json()

    @pytest.mark.asyncio
    async def test_non_artifact_tool_delegates_to_base(self):
        stream = self._make_stream()
        event = make_non_artifact_event("discover_filter_paths", content="filters applied")

        results = [e async for e in stream.handle_function_tool_result(event)]

        # Base class produces a ToolCallResultEvent with the original content
        assert isinstance(results[0], ToolCallResultEvent)
        assert results[0].content == "filters applied"


def _make_request_context(user_text: str = "show subscriptions") -> RequestContext:
    """Create a RequestContext for testing."""
    msg = Message(role=Role.ROLE_USER, parts=[Part(text=user_text)], message_id=str(uuid.uuid4()))
    return RequestContext(call_context=ServerCallContext(), request=SendMessageRequest(message=msg))


async def _collect_events(queue: EventQueue) -> list[Any]:
    """Drain all events from an EventQueue after execute() completes."""
    events = []
    while True:
        try:
            event = queue.queue.get_nowait()
            events.append(event)
        except asyncio.QueueEmpty:
            break
    return events


class TestWFOAgentExecutor:
    @pytest.fixture(autouse=True)
    def _stub_persistence(self):
        """Stub PostgresStatePersistence so executor tests need no real DB.

        load_state -> None, snapshot -> no-op.
        """
        with patch("orchestrator_agent.adapters.a2a.PostgresStatePersistence") as mock_cls:
            instance = mock_cls.return_value
            instance.load_state = AsyncMock(return_value=None)
            instance.snapshot = AsyncMock()
            yield

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_artifact_emitted_as_a2a_artifact(self, _mock_db):
        """Artifacts from agent stream become A2A TaskArtifactUpdateEvents."""
        agent = _agent_mock(
            lambda: mock_event_stream(
                make_non_artifact_event("discover_filter_paths", content="filters applied"),
                make_artifact_event("search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )
        ex = WFOAgentExecutor(agent)

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
        assert len(artifact_events) == 1
        # Structured tool result rides A2A as a DataPart (not raw-JSON text), so it survives
        # for rich/direct clients without polluting text-only consumers.
        data = json_format.MessageToDict(artifact_events[0].artifact.parts[0].data)
        assert data["query_id"] == str(SAMPLE_ARTIFACT.query_id)  # a JSON value in the part, not raw-JSON text
        assert status_events[0].status.state == TaskState.TASK_STATE_WORKING
        assert status_events[-1].status.state == TaskState.TASK_STATE_COMPLETED
        # Completed message is the agent's own (markdown) answer — never raw tool JSON.
        completed_text = status_events[-1].status.message.parts[0].text
        assert completed_text == "Execution completed"

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_passes_user_input_to_agent(self, _mock_db):
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Done")))
        ex = WFOAgentExecutor(agent)

        ctx = _make_request_context("show subscriptions")
        queue = EventQueue()
        await ex.execute(ctx, queue)

        args, _kwargs = agent._last_call
        assert args[0] == "show subscriptions"

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_form_session_restored_from_prior_turn(self, _mock_db):
        """A form being filled spans turns: the validated pages come back with the thread state.

        The user's confirmation arrives on a later message (possibly relayed by another agent), so the
        run's deps must carry the prior turn's ``form_fill``.
        """
        session = FormFillSession(workflow_key="create_node", page_inputs=[{"product": "p-1"}], status="confirming")
        prior = SearchState(form_fill=session)
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Started")))
        with patch("orchestrator_agent.adapters.a2a.PostgresStatePersistence") as mock_cls:
            mock_cls.return_value.load_state = AsyncMock(return_value=prior)
            mock_cls.return_value.snapshot = AsyncMock()
            await WFOAgentExecutor(agent).execute(_make_request_context("yes, go ahead"), EventQueue())

            _args, kwargs = agent._last_call
            assert kwargs["deps"].state.form_fill == session
            # The snapshot at the end of the turn carries the (same) state object onward.
            mock_cls.return_value.snapshot.assert_awaited_once_with(kwargs["deps"].state)

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_a_form_reply_left_by_the_capability_is_the_answer(self, _mock_db):
        """The form-fill capability ends the run with the contract text; the executor delivers exactly that."""

        def claimed(state):
            state.form_fill = FormFillSession(workflow_key="w")
            state.form_reply = Reply("Form page 1 — values needed:\n- speed (required)")

        agent = _agent_doing(claimed, text="Form page 1 — values needed:\n- speed (required)\n\n| table |")
        with patch("orchestrator_agent.adapters.a2a.PostgresStatePersistence") as mock_cls:
            mock_cls.return_value.load_state = AsyncMock(return_value=None)
            mock_cls.return_value.snapshot = AsyncMock()
            queue = EventQueue()
            await WFOAgentExecutor(agent).execute(_make_request_context("create a lightpath"), queue)
            events = await _collect_events(queue)
            last = [e for e in events if isinstance(e, TaskStatusUpdateEvent)][-1]
            assert last.status.state == TaskState.TASK_STATE_COMPLETED
            assert last.status.message.parts[0].text == "Form page 1 — values needed:\n- speed (required)"
            (snapshot_state,), _ = mock_cls.return_value.snapshot.await_args
            assert snapshot_state.form_fill.workflow_key == "w"
            assert "form_reply" not in snapshot_state.model_dump(mode="json")  # transient

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_without_a_form_reply_the_models_text_is_the_answer(self, _mock_db):
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("model answer")))
        queue = EventQueue()
        await WFOAgentExecutor(agent).execute(_make_request_context("how many subscriptions"), queue)
        events = await _collect_events(queue)
        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.message.parts[0].text == "model answer"
        agent.run_stream_events.assert_called_once()

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_completed_with_text_output(self, _mock_db):
        """Text-only output results in completed status with message."""
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Done")))
        ex = WFOAgentExecutor(agent)

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.state == TaskState.TASK_STATE_COMPLETED
        assert status_events[-1].status.message.parts[0].text == "Done"

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_no_output_yields_no_results(self, _mock_db):
        agent = _agent_mock(lambda: mock_event_stream())
        ex = WFOAgentExecutor(agent)

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.state == TaskState.TASK_STATE_COMPLETED
        assert status_events[-1].status.message.parts[0].text == NO_RESULTS

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_failed_on_error(self, _mock_db):
        """Exception during execution results in failed status."""

        def failing_stream():
            async def gen():
                raise RuntimeError("boom")
                yield  # noqa: F401 — makes this an async generator

            return gen()

        agent = _agent_mock(failing_stream)
        ex = WFOAgentExecutor(agent)

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.state == TaskState.TASK_STATE_FAILED

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_cancel(self, _mock_db):
        ex = WFOAgentExecutor(_agent_mock(lambda: mock_event_stream()))
        ctx = _make_request_context()
        queue = EventQueue()
        await ex.cancel(ctx, queue)
        events = await _collect_events(queue)

        assert len(events) == 1
        assert isinstance(events[0], TaskStatusUpdateEvent)
        assert events[0].status.state == TaskState.TASK_STATE_CANCELED


class TestWFOAgentExecutorHITL:
    """kagent's human-in-the-loop extension: stops become input-required pauses on one task, answers resume it."""

    URI = HITL_EXTENSION_URI

    @staticmethod
    def _context(text, *, metadata=None, task_id="task-1", extension=True):
        msg = Message(role=Role.ROLE_USER, parts=[Part(text=text)], message_id=str(uuid.uuid4()), metadata=metadata)
        call_context = ServerCallContext(requested_extensions={TestWFOAgentExecutorHITL.URI} if extension else set())
        return RequestContext(
            call_context=call_context, request=SendMessageRequest(message=msg), task_id=task_id, context_id="ctx-1"
        )

    @staticmethod
    def _persistence(prior=None):
        patcher = patch("orchestrator_agent.adapters.a2a.PostgresStatePersistence")
        mock_cls = patcher.start()
        mock_cls.return_value.load_state = AsyncMock(return_value=prior)
        mock_cls.return_value.snapshot = AsyncMock()
        return patcher, mock_cls

    def test_card_declares_the_extension(self):
        adapter = A2AAdapter(_agent_mock(lambda: mock_event_stream()), url="http://x/")
        card = adapter.agent_card
        assert [e.uri for e in card.capabilities.extensions] == [self.URI]
        assert card.capabilities.extensions[0].required is False
        assert [i.protocol_version for i in card.supported_interfaces] == ["1.0"]

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_a_stop_pauses_the_task_with_an_ask_user_request(self, _mock_db):
        from orchestrator_agent.state import AskField, FormFillSession, Reply

        reply = Reply(
            "Form page 2 — values needed:",
            ask=[AskField(name="redundancy", question="Redundancy?", choices=["protected", "unprotected"])],
        )

        def stopped(state):
            state.form_fill = FormFillSession(workflow_key="w", status="gathering")
            state.form_reply = reply

        patcher, mock_cls = self._persistence()
        try:
            queue = EventQueue()
            ctx = self._context("create a lightpath")
            await WFOAgentExecutor(_agent_doing(stopped, text=reply.text)).execute(ctx, queue)
            events = await _collect_events(queue)
            last = [e for e in events if isinstance(e, TaskStatusUpdateEvent)][-1]
            assert last.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
            assert last.status.message.parts[0].text == reply.text
            payload = json_format.MessageToDict(last.status.message.metadata)[self.URI]
            assert payload["type"] == "ask_user_request" and payload["questions"] == [
                {"question": "Redundancy?", "choices": ["protected", "unprotected"], "multiple": False}
            ]
            assert list(last.status.message.extensions) == [self.URI]
            # The session remembers what was asked (keyed by the request id) and its task, before the pause.
            (snapshot_state,), _ = mock_cls.return_value.snapshot.await_args
            assert snapshot_state.form_fill.hitl_request == {
                "id": payload["id"],
                "kind": "ask",
                "questions": [{"field": "redundancy", "multiple": False, "options": None}],
            }
            assert mock_cls.call_args.kwargs["thread_id"] == "ctx-1"  # memory stays keyed on the context
            assert snapshot_state.form_fill.task_id == "task-1"  # ...the form session knows its task
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_an_ask_user_response_becomes_the_text_the_skill_reads(self, _mock_db):
        from orchestrator_agent.state import FormFillSession, SearchState

        prior = SearchState(
            form_fill=FormFillSession(
                workflow_key="w",
                status="gathering",
                hitl_request={
                    "id": "req-9",
                    "kind": "ask",
                    "questions": [{"field": "redundancy"}, {"field": "ticket_id"}],
                },
            )
        )
        metadata = {
            self.URI: {
                "type": "ask_user_response",
                "id": "req-9",
                "answers": [{"answer": ["protected"]}, {"answer": []}],
            }
        }
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("ok")))
        patcher, _ = self._persistence(prior)
        try:
            await WFOAgentExecutor(agent).execute(
                self._context("Human input supplied", metadata=metadata), EventQueue()
            )
            args, kwargs = agent._last_call
            assert args[0] == "Human input supplied" and kwargs["deps"].state.user_input == args[0]
            assert kwargs["deps"].state.form_values == {"redundancy": "protected"}  # an empty answer sends nothing
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    @pytest.mark.parametrize(
        "approval, expected_text, expected_decision",
        [
            ({"id": "req-5", "approved": True}, "Human input supplied", "start"),
            ({"id": "req-5", "approved": False, "rejection_reason": "not now"}, "Human input supplied", "cancel"),
            ({"id": "req-5", "approved": False, "rejection_reason": '{"speed": "1000"}'}, "Human input supplied", None),
        ],
    )
    async def test_a_tool_approval_response_is_a_decision_or_a_correction(
        self, _mock_db, approval, expected_text, expected_decision
    ):
        from orchestrator_agent.state import FormFillSession, SearchState

        prior = SearchState(
            form_fill=FormFillSession(
                workflow_key="w", status="confirming", hitl_request={"id": "req-5", "kind": "approval"}
            )
        )
        metadata = {self.URI: {"type": "tool_approval_response", "approvals": [approval]}}
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("ok")))
        patcher, _ = self._persistence(prior)
        try:
            await WFOAgentExecutor(agent).execute(
                self._context("Human input supplied", metadata=metadata), EventQueue()
            )
            args, kwargs = agent._last_call
            assert args[0] == expected_text and kwargs["deps"].state.form_decision == expected_decision
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_a_session_from_another_task_is_not_continued_but_survives(self, _mock_db):
        from orchestrator_agent.state import FormFillSession, SearchState

        paused = FormFillSession(
            workflow_key="w", status="gathering", task_id="task-OLD", hitl_request={"id": "r", "kind": "ask"}
        )
        prior = SearchState(form_fill=paused)
        seen = []
        agent = _agent_doing(lambda state: seen.append(state.form_fill), text="ok")
        patcher, mock_cls = self._persistence(prior)
        try:
            await WFOAgentExecutor(agent).execute(self._context("list subscriptions", task_id="task-NEW"), EventQueue())
            assert seen == [None]  # the old task's form does not leak into the new task's run...
            (snapshot_state,), _ = mock_cls.return_value.snapshot.await_args
            assert snapshot_state.form_fill == paused  # ...and is still there for the human's answer on the old task
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_without_the_extension_a_stop_is_plain_completed_text(self, _mock_db):
        from orchestrator_agent.state import AskField, FormFillSession, Reply

        def stopped(state):
            state.form_fill = FormFillSession(workflow_key="w")
            state.form_reply = Reply("values needed: ...", ask=[AskField(name="x", question="X?")])

        patcher, mock_cls = self._persistence()
        try:
            queue = EventQueue()
            await WFOAgentExecutor(_agent_doing(stopped, text="values needed: ...")).execute(
                self._context("create", extension=False), queue
            )
            events = await _collect_events(queue)
            last = [e for e in events if isinstance(e, TaskStatusUpdateEvent)][-1]
            assert last.status.state == TaskState.TASK_STATE_COMPLETED
            assert not json_format.MessageToDict(last.status.message.metadata)
            assert mock_cls.call_args.kwargs["thread_id"] == "ctx-1"
        finally:
            patcher.stop()


class TestA2AEndpoint:
    """HTTP-level tests for the A2A adapter via a2a-sdk."""

    @pytest.fixture(autouse=True)
    def _stub_persistence(self):
        """Stub PostgresStatePersistence so HTTP-level tests need no real DB.

        load_state -> None, snapshot -> no-op.
        """
        with patch("orchestrator_agent.adapters.a2a.PostgresStatePersistence") as mock_cls:
            instance = mock_cls.return_value
            instance.load_state = AsyncMock(return_value=None)
            instance.snapshot = AsyncMock()
            yield

    @pytest.fixture
    async def a2a_app(self):
        from fastapi import FastAPI

        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Done")))
        adapter = A2AAdapter(agent, url="http://localhost:8080/")
        app = FastAPI()
        adapter.add_routes(app)
        yield type("A2AAppFixture", (), {"app": app, "agent": agent, "executor": adapter.executor})

    @staticmethod
    def _jsonrpc_request(method: str, user_text: str = "show subscriptions", req_id: int = 1) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": {
                "message": {"role": "ROLE_USER", "parts": [{"text": user_text}], "messageId": str(uuid.uuid4())}
            },
        }

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_send_message_returns_a_completed_task_with_artifacts(self, _mock_db, a2a_app):
        a2a_app.executor.agent = _agent_mock(
            lambda: mock_event_stream(
                make_artifact_event("search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )
        transport = httpx.ASGITransport(app=a2a_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/", json=self._jsonrpc_request("SendMessage"), headers={"A2A-Version": "1.0"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0" and body["id"] == 1
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert len(task["artifacts"]) >= 1
        assert task["status"]["message"]["parts"][0]["text"] == "Execution completed"

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_send_streaming_message_ends_completed(self, _mock_db, a2a_app):
        transport = httpx.ASGITransport(app=a2a_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/", json=self._jsonrpc_request("SendStreamingMessage"), headers={"A2A-Version": "1.0"}
            )
        assert resp.status_code == 200
        events = [json.loads(line[6:]) for line in resp.text.strip().split("\n") if line.startswith("data: ")]
        states = [
            e["result"]["statusUpdate"]["status"]["state"] for e in events if "statusUpdate" in e.get("result", {})
        ]
        assert "TASK_STATE_WORKING" in states and states[-1] == "TASK_STATE_COMPLETED"

    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self, a2a_app):
        transport = httpx.ASGITransport(app=a2a_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "WFO Agent" and card["capabilities"]["streaming"] is True
        assert [i["protocolVersion"] for i in card["supportedInterfaces"]] == ["1.0"]
        assert len(card["skills"]) == len(A2A_SKILLS)


def _run_agent_mock(output: str = "", messages: list | None = None) -> MagicMock:
    """An agent mock whose `run(...)` returns a result with the given output and `all_messages()`.

    It's also an async context manager. The MCP worker uses `agent.run`, not the streaming path.
    """
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    result = MagicMock(output=output)
    result.all_messages.return_value = messages or []
    agent.run = AsyncMock(return_value=result)
    return agent


class TestMCPWorker:
    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_returns_prose_when_no_artifact_data(self, _mock_db):
        # No artifact-bearing tool result -> just the agent's final answer.
        agent = _run_agent_mock(output="There are 7 active subscriptions.")

        out = await MCPWorker(agent=agent).run("show subscriptions")

        assert out == "There are 7 active subscriptions."
        assert agent.run.call_args.args[0] == "show subscriptions"

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_appends_artifact_payload_as_json(self, _mock_db):
        # An artifact-bearing tool result -> its payload JSON is appended after the prose.
        part = ToolReturnPart(tool_name="search", content={"query_id": "q-1", "returned": 2}, tool_call_id="c1")
        part.metadata = QueryArtifact(description="d", query_id="q-1", total_results=2)
        agent = _run_agent_mock(output="Found 2.", messages=[SimpleNamespace(parts=[part])])

        out = await MCPWorker(agent=agent).run("find subs")

        assert out.startswith("Found 2.")
        assert "q-1" in out  # the structured payload rides along as JSON

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_exception_propagates(self, _mock_db):
        """Exception during the agent run is re-raised after logging."""
        agent = _run_agent_mock()
        agent.run.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await MCPWorker(agent=agent).run("find subs")


class TestMCPAppInit:
    @patch("orchestrator_agent.adapters.mcp.db")
    def test_init_creates_worker_and_app(self, _mock_db):
        agent = MagicMock()
        app = MCPApp(agent)

        assert app.worker.agent is agent
        assert app.app is not None
        assert app.server is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool", "query"),
        [
            ("search", "find active subs"),  # auto-derived from the search plugin
            ("aggregate", "count by product"),  # auto-derived from the aggregate plugin
            ("entity", "show subscription abc"),  # auto-derived from the entity plugin
            ("export", "export the last results"),  # auto-derived from the export plugin
            ("ask", "what is happening?"),  # generic catch-all
        ],
    )
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_tool_delegates_to_worker(self, _mock_db, tool, query):
        agent = MagicMock()
        app = MCPApp(agent)
        app.worker.run = AsyncMock(return_value="result")

        await app.server.call_tool(tool, {"query": query})

        app.worker.run.assert_called_once()
        assert app.worker.run.call_args.args[0] == query

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_tools_auto_registered_one_per_plugin(self, _mock_db):
        # The MCP surface is derived from the plugin set — one tool per advertised plugin, plus `ask`.
        from orchestrator_agent.capabilities import load_plugin_specs

        app = MCPApp(MagicMock())
        names = {t.name for t in await app.server.list_tools()}
        advertised = {s.id for s in load_plugin_specs() if s.advertise}
        assert advertised <= names  # every advertised plugin is a tool
        assert "ask" in names


class TestMCPAppLifecycle:
    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_aenter_aexit(self, _mock_db):
        agent = MagicMock()
        app = MCPApp(agent)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=None)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(app.server.session_manager, "run", return_value=mock_cm):
            async with app as ctx:
                assert ctx is app
            mock_cm.__aenter__.assert_called_once()
            mock_cm.__aexit__.assert_called_once()


class TestAGUIAdapterBuildEventStream:
    def test_build_event_stream_returns_agui_event_stream(self):
        agent = MagicMock()
        adapter = _AGUIAdapter(agent=agent, run_input=minimal_run_input())
        stream = adapter.build_event_stream()

        assert isinstance(stream, AGUIEventStream)


class TestAGUIWorkerStaticMethods:
    _RUN_ID = "00000000-0000-0000-0000-000000000001"

    @pytest.mark.parametrize(
        ("messages", "expected_user_input"),
        [
            ([UserMessage(id="m1", role="user", content="find active subs")], "find active subs"),
            ([], ""),
        ],
    )
    def test_prepare_run_input(self, messages, expected_user_input):
        run_input = RunAgentInput(
            thread_id="t1",
            run_id=self._RUN_ID,
            state={"existing_key": "val"},
            messages=messages,
            tools=[],
            context=[],
            forwarded_props={},
        )
        result = AGUIWorker._prepare_run_input(run_input)

        assert result.state["user_input"] == expected_user_input
        assert result.state["existing_key"] == "val"
        assert result.thread_id == "t1"
        assert result.run_id == run_input.run_id

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            ([], ""),
            ([UserMessage(id="m1", role="user", content="only message")], "only message"),
            (
                [
                    UserMessage(id="m1", role="user", content="first message"),
                    UserMessage(id="m2", role="user", content="second message"),
                ],
                "second message",
            ),
        ],
    )
    def test_extract_user_input(self, messages, expected):
        run_input = RunAgentInput(
            thread_id="t1",
            run_id=self._RUN_ID,
            state={},
            messages=messages,
            tools=[],
            context=[],
            forwarded_props={},
        )
        assert AGUIWorker._extract_user_input(run_input) == expected


class TestAGUIWorkerRunRequest:
    def _make_run_input(self, msg: str = "find subs") -> RunAgentInput:
        return RunAgentInput(
            thread_id="t-thread",
            run_id="00000000-0000-0000-0000-000000000042",
            state={},
            messages=[UserMessage(id="m1", role="user", content=msg)],
            tools=[],
            context=[],
            forwarded_props={},
        )

    def _agent(self) -> MagicMock:
        agent = MagicMock()
        agent.__aenter__ = AsyncMock(return_value=agent)
        agent.__aexit__ = AsyncMock(return_value=False)
        return agent

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_new_run_streams_events(self, mock_adapter_cls, mock_persistence_cls):
        agent = self._agent()
        db_session = MagicMock()
        db_session.get.return_value = None  # No existing run

        mock_persistence = AsyncMock()
        mock_persistence.load_state.return_value = None
        mock_persistence_cls.return_value = mock_persistence

        async def mock_sse_stream():
            yield "data: event1\n\n"
            yield "data: event2\n\n"

        mock_adapter = MagicMock()
        mock_adapter.run_stream.return_value = mock_event_stream()
        mock_adapter.encode_stream.return_value = mock_sse_stream()
        mock_adapter_cls.return_value = mock_adapter

        stream = await AGUIWorker.run_request(agent, self._make_run_input(), db_session)
        chunks = [c async for c in stream]

        assert chunks == ["data: event1\n\n", "data: event2\n\n"]
        db_session.add.assert_called_once()
        db_session.commit.assert_called()
        mock_persistence.snapshot.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_existing_run_skips_insert(self, mock_adapter_cls, mock_persistence_cls):
        from orchestrator.core.db.models import AgentRunTable

        agent = self._agent()
        db_session = MagicMock()
        existing_run = MagicMock(spec=AgentRunTable)
        db_session.get.return_value = existing_run  # Run already exists

        mock_persistence = AsyncMock()
        mock_persistence.load_state.return_value = None
        mock_persistence_cls.return_value = mock_persistence

        async def empty_stream():
            return
            yield  # noqa: F401

        mock_adapter = MagicMock()
        mock_adapter.run_stream.return_value = mock_event_stream()
        mock_adapter.encode_stream.return_value = empty_stream()
        mock_adapter_cls.return_value = mock_adapter

        stream = await AGUIWorker.run_request(agent, self._make_run_input(), db_session)
        _ = [c async for c in stream]

        db_session.add.assert_not_called()

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_loads_previous_state(self, mock_adapter_cls, mock_persistence_cls):
        agent = self._agent()
        db_session = MagicMock()
        db_session.get.return_value = None

        previous_state = SearchState(user_input="old query")
        mock_persistence = AsyncMock()
        mock_persistence.load_state.return_value = previous_state
        mock_persistence_cls.return_value = mock_persistence

        async def empty_stream():
            return
            yield  # noqa: F401

        mock_adapter = MagicMock()
        mock_adapter.run_stream.return_value = mock_event_stream()
        mock_adapter.encode_stream.return_value = empty_stream()
        mock_adapter_cls.return_value = mock_adapter

        stream = await AGUIWorker.run_request(agent, self._make_run_input("new query"), db_session)
        _ = [c async for c in stream]

        # The previous state's user_input is updated to the new request.
        call_kwargs = mock_adapter_cls.call_args.kwargs
        initial_state = call_kwargs["run_input"].state
        assert initial_state["user_input"] == "new query"

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_rollback_on_stream_error(self, mock_adapter_cls, mock_persistence_cls):
        agent = self._agent()
        db_session = MagicMock()
        db_session.get.return_value = None

        mock_persistence = AsyncMock()
        mock_persistence.load_state.return_value = None
        mock_persistence_cls.return_value = mock_persistence

        async def failing_stream():
            yield "data: partial\n\n"
            raise RuntimeError("stream error")

        mock_adapter = MagicMock()
        mock_adapter.run_stream.return_value = mock_event_stream()
        mock_adapter.encode_stream.return_value = failing_stream()
        mock_adapter_cls.return_value = mock_adapter

        stream = await AGUIWorker.run_request(agent, self._make_run_input(), db_session)
        with pytest.raises(RuntimeError, match="stream error"):
            _ = [c async for c in stream]

        db_session.rollback.assert_called_once()
