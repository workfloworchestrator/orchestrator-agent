"""Adapter output tests — verify each protocol adapter transforms agent events correctly.

We mock `agent.run_stream_events()` to yield pre-built event sequences.
No LLM calls, no DB calls — just adapter transformation logic.
"""

from __future__ import annotations

import asyncio
import json
import uuid
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

from orchestrator_agent.adapters.a2a import A2A_SKILLS, WFOAgentExecutor, _build_state_fallback
from orchestrator_agent.adapters.ag_ui import AGUIEventStream, AGUIWorker, _AGUIAdapter
from orchestrator_agent.adapters.mcp import MCPApp, MCPWorker
from orchestrator_agent.adapters.stream import NO_RESULTS, collect_stream_output
from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.events import AgentStepActiveEvent
from orchestrator_agent.memory import ToolStep
from orchestrator_agent.state import SearchState, TaskAction

from .conftest import (
    make_artifact_event,
    make_non_artifact_event,
    make_step_event,
    make_text_result_event,
    minimal_run_input,
    mock_event_stream,
)

SAMPLE_ARTIFACT = QueryArtifact(
    description="Found 5 subscriptions",
    query_id="q-123",
    total_results=5,
)


class TestCollectStreamOutput:
    @pytest.mark.asyncio
    async def test_artifacts_take_priority_over_text(self):
        stream = mock_event_stream(
            make_artifact_event("run_search", SAMPLE_ARTIFACT),
            make_text_result_event("Execution completed"),
        )
        result = await collect_stream_output(stream)
        assert result == SAMPLE_ARTIFACT.model_dump_json()

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        result = await collect_stream_output(mock_event_stream())
        assert result == NO_RESULTS


class TestAGUIEventStream:
    def _make_stream(self) -> AGUIEventStream:
        return AGUIEventStream(run_input=minimal_run_input())

    @pytest.mark.asyncio
    async def test_artifact_becomes_lightweight_json(self):
        stream = self._make_stream()
        event = make_artifact_event("run_search", SAMPLE_ARTIFACT)

        results = [e async for e in stream.handle_function_tool_result(event)]

        assert isinstance(results[0], ToolCallResultEvent)
        assert results[0].content == SAMPLE_ARTIFACT.model_dump_json()

    @pytest.mark.asyncio
    async def test_custom_event_passes_through(self):
        stream = self._make_stream()
        results = [e async for e in stream.handle_event(make_step_event("planning"))]

        assert isinstance(results[0], AgentStepActiveEvent)
        assert results[0].value.get("step") == "planning"

    @pytest.mark.asyncio
    async def test_non_artifact_tool_delegates_to_base(self):
        stream = self._make_stream()
        event = make_non_artifact_event("set_filters", content="filters applied")

        results = [e async for e in stream.handle_function_tool_result(event)]

        # Base class produces a ToolCallResultEvent with the original content
        assert isinstance(results[0], ToolCallResultEvent)
        assert results[0].content == "filters applied"


def _make_request_context(user_text: str = "show subscriptions", skill_id: str | None = None) -> RequestContext:
    """Create a RequestContext for testing."""
    metadata = {}
    if skill_id:
        metadata["skill_id"] = skill_id
    msg = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=user_text))],
        message_id=str(uuid.uuid4()),
        metadata=metadata or None,
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
    @pytest.fixture
    def executor(self):
        agent = AsyncMock()
        return WFOAgentExecutor(agent), agent

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_artifact_emitted_as_a2a_artifact(self, _mock_db, executor):
        """Artifacts from agent stream become A2A TaskArtifactUpdateEvents."""
        ex, agent = executor
        agent.run_stream_events = MagicMock(
            return_value=mock_event_stream(
                make_non_artifact_event("set_filters", content="filters applied"),
                make_artifact_event("run_search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        # Should have: working, artifact, completed
        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
        assert len(artifact_events) == 1
        assert artifact_events[0].artifact.parts[0].root.text == SAMPLE_ARTIFACT.model_dump_json()
        assert status_events[0].status.state == TaskState.working
        assert status_events[-1].status.state == TaskState.completed
        # Completed message should contain artifact content, not generic "Execution completed"
        completed_text = status_events[-1].status.message.parts[0].root.text
        assert completed_text == SAMPLE_ARTIFACT.model_dump_json()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("skill_id", "expected"), [("search", TaskAction.SEARCH), ("nonexistent", None)])
    @patch("orchestrator.db.db")
    async def test_skill_id_routing(self, _mock_db, executor, skill_id, expected):
        ex, agent = executor
        agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

        ctx = _make_request_context(skill_id=skill_id)
        queue = EventQueue()
        await ex.execute(ctx, queue)

        assert agent.run_stream_events.call_args.kwargs["target_action"] == expected

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_completed_with_text_output(self, _mock_db, executor):
        """Text-only output results in completed status with message."""
        ex, agent = executor
        agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.state == TaskState.completed
        assert status_events[-1].status.message.parts[0].root.text == "Done"

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_failed_on_error(self, _mock_db, executor):
        """Exception during execution results in failed status."""
        ex, agent = executor

        async def failing_stream(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # noqa: F401 — makes this an async generator

        agent.run_stream_events = MagicMock(return_value=failing_stream())

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert status_events[-1].status.state == TaskState.failed

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_cancel(self, _mock_db, executor):
        ex, _agent = executor
        ctx = _make_request_context()
        queue = EventQueue()
        await ex.cancel(ctx, queue)
        events = await _collect_events(queue)

        assert len(events) == 1
        assert isinstance(events[0], TaskStatusUpdateEvent)
        assert events[0].status.state == TaskState.canceled


class TestWFOAgentExecutorStateFallback:
    """Tests for state-based fallback when event stream yields no useful output."""

    @pytest.fixture
    def executor(self):
        agent = AsyncMock()
        return WFOAgentExecutor(agent), agent

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_state_fallback_over_execution_completed(self, _mock_db, executor):
        """When agent yields only 'Execution completed' but state has tool steps, use state."""
        ex, agent = executor

        async def stream_with_state_mutation(*args, **kwargs):
            deps = kwargs.get("deps")
            if deps:
                deps.state.memory.start_turn(deps.state.user_input)
                deps.state.memory.start_step("Search")
                deps.state.memory.record_tool_step(
                    ToolStep(step_type="run_search", description="Searched 5 subscriptions")
                )
                deps.state.memory.finish_step()
                deps.state.memory.complete_turn(assistant_answer="Searched 5 subscriptions.")
            yield make_text_result_event("Execution completed")

        agent.run_stream_events = MagicMock(side_effect=stream_with_state_mutation)

        ctx = _make_request_context()
        queue = EventQueue()
        await ex.execute(ctx, queue)
        events = await _collect_events(queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        completed = status_events[-1]
        assert completed.status.state == TaskState.completed
        message_text = completed.status.message.parts[0].root.text
        assert "Searched 5 subscriptions" in message_text
        assert message_text != "Execution completed"


def _make_state_with_tool_steps(*descriptions: str) -> SearchState:
    """Create a SearchState with completed turn containing tool steps."""
    state = SearchState(user_input="test query")
    state.memory.start_turn("test query")
    state.memory.start_step("Search")
    for desc in descriptions:
        state.memory.record_tool_step(ToolStep(step_type="run_search", description=desc))
    state.memory.finish_step()
    state.memory.complete_turn(assistant_answer="done")
    return state


class TestStateFallback:
    """Tests for state-based fallback when event stream yields no useful output."""

    def test_build_state_fallback_with_descriptions(self):
        state = _make_state_with_tool_steps("Searched 5 subscriptions", "Fetched details for sub-1")
        result = _build_state_fallback(state)
        assert result == "Searched 5 subscriptions. Fetched details for sub-1."

    def test_build_state_fallback_empty(self):
        state = SearchState(user_input="test")
        result = _build_state_fallback(state)
        assert result == "Execution completed"


class TestA2AEndpoint:
    """HTTP-level tests for the A2A adapter via a2a-sdk."""

    @pytest.fixture
    async def a2a_app(self):
        agent = AsyncMock()
        executor = WFOAgentExecutor(agent)
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        )
        agent_card = AgentCard(
            name="Test Agent",
            description="Test",
            url="http://localhost:8000",
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
    @patch("orchestrator.db.db")
    async def test_message_send_endpoint_returns_completed(self, _mock_db, a2a_app):
        """HTTP message/send returns a JSON-RPC response with completed task and artifacts."""
        a2a_app.agent.run_stream_events = MagicMock(
            return_value=mock_event_stream(
                make_artifact_event("run_search", SAMPLE_ARTIFACT),
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
    @patch("orchestrator.db.db")
    async def test_message_stream_endpoint_returns_completed(self, _mock_db, a2a_app):
        """HTTP message/stream returns SSE events ending with completed status."""
        a2a_app.agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

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
    async def test_passes_query_and_action_to_agent(self, _mock_db):
        agent = MagicMock()
        agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

        worker = MCPWorker(agent=agent)
        await worker.run_skill("show subscriptions", target_action=TaskAction.SEARCH)

        call_kwargs = agent.run_stream_events.call_args.kwargs
        assert call_kwargs["deps"].state.user_input == "show subscriptions"
        assert call_kwargs["target_action"] == TaskAction.SEARCH

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_exception_propagates(self, _mock_db):
        """Exception in agent stream is re-raised after logging."""
        agent = MagicMock()

        async def failing_stream(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # noqa: F401

        agent.run_stream_events = MagicMock(return_value=failing_stream())

        worker = MCPWorker(agent=agent)
        with pytest.raises(RuntimeError, match="boom"):
            await worker.run_skill("find subs")


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
        ("tool", "query", "expected_action"),
        [
            ("search", "find active subs", TaskAction.SEARCH),
            ("aggregate", "count by product", TaskAction.AGGREGATION),
            ("ask", "what is happening?", None),
        ],
    )
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_tool_delegates_to_worker(self, _mock_db, tool, query, expected_action):
        agent = MagicMock()
        app = MCPApp(agent)
        app.worker.run_skill = AsyncMock(return_value="result")

        await app.server.call_tool(tool, {"query": query})

        app.worker.run_skill.assert_called_once()
        ca = app.worker.run_skill.call_args
        assert ca.args[0] == query
        actual_action = ca.args[1] if len(ca.args) > 1 else ca.kwargs.get("target_action")
        assert actual_action == expected_action

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.mcp.db")
    async def test_tool_get_entity_details_delegates_to_worker(self, _mock_db):
        import uuid as _uuid

        agent = MagicMock()
        app = MCPApp(agent)
        app.worker.run_skill = AsyncMock(return_value="details result")

        entity_id = _uuid.uuid4()
        await app.server.call_tool("get_entity_details", {"entity_type": "SUBSCRIPTION", "entity_id": str(entity_id)})

        call_kwargs = app.worker.run_skill.call_args
        assert call_kwargs.args[1] == TaskAction.RESULT_ACTIONS
        assert "SUBSCRIPTION" in call_kwargs.args[0]
        assert str(entity_id) in call_kwargs.args[0]


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


class TestAGUIEventStreamSuperDelegate:
    @pytest.mark.asyncio
    async def test_non_custom_event_delegates_to_super(self):
        """Non-CustomEvent NativeEvents fall through to the base class handler."""
        stream = AGUIEventStream(run_input=minimal_run_input())
        event = make_text_result_event("Done")  # AgentRunResultEvent — not a CustomEvent

        # Should not raise; base class may or may not yield events for this type
        results = [e async for e in stream.handle_event(event)]
        assert isinstance(results, list)


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

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_new_run_streams_events(self, mock_adapter_cls, mock_persistence_cls):
        agent = MagicMock()
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

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_existing_run_skips_insert(self, mock_adapter_cls, mock_persistence_cls):
        from orchestrator.db.models import AgentRunTable

        agent = MagicMock()
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
        agent = MagicMock()
        db_session = MagicMock()
        db_session.get.return_value = None

        previous_state = SearchState(user_input="old query")
        mock_persistence = AsyncMock()
        mock_persistence.load_state.return_value = previous_state
        mock_persistence_cls.return_value = mock_persistence

        captured_deps = {}

        async def empty_stream():
            return
            yield  # noqa: F401

        def capture_run_stream(deps):
            captured_deps["deps"] = deps
            return mock_event_stream()

        mock_adapter = MagicMock()
        mock_adapter.run_stream.side_effect = capture_run_stream
        mock_adapter.encode_stream.return_value = empty_stream()
        mock_adapter_cls.return_value = mock_adapter

        stream = await AGUIWorker.run_request(agent, self._make_run_input("new query"), db_session)
        _ = [c async for c in stream]

        # Adapter was constructed; the previous state's user_input is updated
        call_kwargs = mock_adapter_cls.call_args.kwargs
        initial_state = call_kwargs["run_input"].state
        assert initial_state["user_input"] == "new query"

    @pytest.mark.asyncio
    @patch("orchestrator_agent.adapters.ag_ui.PostgresStatePersistence")
    @patch("orchestrator_agent.adapters.ag_ui._AGUIAdapter")
    async def test_run_request_rollback_on_stream_error(self, mock_adapter_cls, mock_persistence_cls):
        agent = MagicMock()
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
