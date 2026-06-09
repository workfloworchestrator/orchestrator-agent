"""Adapter output tests — verify each protocol adapter transforms agent events correctly.

We mock `agent.run_stream_events()` to yield pre-built event sequences.
No LLM calls, no DB calls — just adapter transformation logic.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from ag_ui.core import RunAgentInput, ToolCallResultEvent, UserMessage

from orchestrator_agent.adapters.a2a import A2A_SKILLS, NO_RESULTS, WFOAgentExecutor
from orchestrator_agent.adapters.ag_ui import AGUIEventStream, AGUIWorker, _AGUIAdapter
from orchestrator_agent.adapters.mcp import MCPApp, MCPWorker
from orchestrator_agent.adapters.stream import collect_stream_output
from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.state import SearchState

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
    msg = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=user_text))],
        message_id=str(uuid.uuid4()),
    )
    params = MessageSendParams(message=msg)
    return RequestContext(request=params)


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
        assert artifact_events[0].artifact.parts[0].root.data == json.loads(SAMPLE_ARTIFACT.model_dump_json())
        assert status_events[0].status.state == TaskState.working
        assert status_events[-1].status.state == TaskState.completed
        # Completed message is the agent's own (markdown) answer — never raw tool JSON.
        completed_text = status_events[-1].status.message.parts[0].root.text
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
    async def test_completed_with_text_output(self, _mock_db):
        """Text-only output results in completed status with message."""
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Done")))
        ex = WFOAgentExecutor(agent)

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.state == TaskState.completed
        assert status_events[-1].status.message.parts[0].root.text == "Done"

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
        assert status_events[-1].status.state == TaskState.completed
        assert status_events[-1].status.message.parts[0].root.text == NO_RESULTS

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
        assert status_events[-1].status.state == TaskState.failed

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
        assert events[0].status.state == TaskState.canceled


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
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Done")))
        executor = WFOAgentExecutor(agent)
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        )
        agent_card = AgentCard(
            name="Test Agent",
            url="http://localhost:8080",
            description="Test",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=A2A_SKILLS,
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
        )
        a2a = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)
        from fastapi import FastAPI

        app = FastAPI()
        a2a.add_routes_to_app(app)
        yield type("A2AAppFixture", (), {"app": app, "agent": agent, "executor": executor})

    @staticmethod
    def _jsonrpc_request(method: str, user_text: str = "show subscriptions", req_id: int = 1) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": user_text}],
                    "messageId": str(uuid.uuid4()),
                }
            },
        }

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_message_send_endpoint_returns_completed(self, _mock_db, a2a_app):
        """HTTP message/send returns a JSON-RPC response with a completed task and artifacts."""
        a2a_app.executor.agent = _agent_mock(
            lambda: mock_event_stream(
                make_artifact_event("search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )

        transport = httpx.ASGITransport(app=a2a_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/", json=self._jsonrpc_request("message/send"))

        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 1
        result = body["result"]
        assert result["status"]["state"] == "completed"
        assert len(result["artifacts"]) >= 1

    @pytest.mark.asyncio
    @patch("orchestrator.core.db.db")
    async def test_message_stream_endpoint_returns_completed(self, _mock_db, a2a_app):
        """HTTP message/stream returns SSE events ending with completed status."""
        transport = httpx.ASGITransport(app=a2a_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/", json=self._jsonrpc_request("message/stream"))

        assert resp.status_code == 200
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) >= 2
        states = [e["result"]["status"]["state"] for e in events if "status" in e.get("result", {})]
        assert "working" in states
        assert "completed" in states

    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self, a2a_app):
        """GET /.well-known/agent.json returns the agent card."""
        transport = httpx.ASGITransport(app=a2a_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "Test Agent"
        assert card["capabilities"]["streaming"] is True
        assert len(card["skills"]) == len(A2A_SKILLS)


class TestMCPWorker:
    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_passes_query_to_agent(self, _mock_db):
        agent = _agent_mock(lambda: mock_event_stream(make_text_result_event("Done")))

        worker = MCPWorker(agent=agent)
        await worker.run("show subscriptions")

        args, _kwargs = agent._last_call
        assert args[0] == "show subscriptions"

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_exception_propagates(self, _mock_db):
        """Exception in agent stream is re-raised after logging."""

        def failing_stream():
            async def gen():
                raise RuntimeError("boom")
                yield  # noqa: F401

            return gen()

        agent = _agent_mock(failing_stream)

        worker = MCPWorker(agent=agent)
        with pytest.raises(RuntimeError, match="boom"):
            await worker.run("find subs")


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
            ("search", "find active subs"),
            ("aggregate", "count by product"),
            ("ask", "what is happening?"),
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
    async def test_tool_get_entity_details_delegates_to_worker(self, _mock_db):
        import uuid as _uuid

        agent = MagicMock()
        app = MCPApp(agent)
        app.worker.run = AsyncMock(return_value="details result")

        entity_id = _uuid.uuid4()
        await app.server.call_tool("get_entity_details", {"entity_type": "SUBSCRIPTION", "entity_id": str(entity_id)})

        query = app.worker.run.call_args.args[0]
        assert "SUBSCRIPTION" in query
        assert str(entity_id) in query


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
