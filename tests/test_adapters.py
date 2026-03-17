"""Adapter output tests — verify each protocol adapter transforms agent events correctly.

We mock `agent.run_stream_events()` to yield pre-built event sequences.
No LLM calls, no DB calls — just adapter transformation logic.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from ag_ui.core import ToolCallResultEvent
from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Message, TaskSendParams
from fasta2a.schema import TextPart as A2ATextPart
from fasta2a.storage import InMemoryStorage

from orchestrator_agent.adapters.a2a import A2AWorker, StreamingFastA2A, _build_state_fallback
from orchestrator_agent.adapters.ag_ui import AGUIEventStream
from orchestrator_agent.adapters.mcp import MCPWorker
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
    parse_sse_events,
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


class TestA2AWorker:
    @pytest.fixture
    def a2a(self):
        """Provides a ready-to-use A2AWorker with mocked agent and DB."""
        storage = InMemoryStorage()
        agent = AsyncMock()
        worker = A2AWorker(agent=agent, broker=InMemoryBroker(), storage=storage)

        async def submit(user_text="show subscriptions", skill_id=None):
            msg: Message = Message(
                role="user",
                parts=[A2ATextPart(kind="text", text=user_text)],
                kind="message",
                message_id=str(uuid.uuid4()),
            )
            context_id = f"ctx-{uuid.uuid4()}"
            task = await storage.submit_task(context_id, msg)
            metadata = {"skill_id": skill_id} if skill_id else {}
            return TaskSendParams(
                id=task["id"],
                context_id=context_id,
                message={**msg, "metadata": metadata},
            )

        return type("A2AFixture", (), {"worker": worker, "agent": agent, "storage": storage, "submit": submit})

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_task_output_contains_artifact_not_text(self, _mock_db, a2a):
        a2a.agent.run_stream_events = MagicMock(
            return_value=mock_event_stream(
                make_non_artifact_event("set_filters", content="filters applied"),
                make_artifact_event("run_search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )

        params = await a2a.submit()
        await a2a.worker.run_task(params)

        task = await a2a.storage.load_task(params["id"])
        agent_msgs = [m for m in task["history"] if m["role"] == "agent"]
        assert len(agent_msgs) == 1
        assert agent_msgs[0]["parts"][0]["text"] == SAMPLE_ARTIFACT.model_dump_json()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("skill_id", "expected"), [("search", TaskAction.SEARCH), ("nonexistent", None)])
    @patch("orchestrator.db.db")
    async def test_skill_id_routing(self, _mock_db, a2a, skill_id, expected):
        a2a.agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))
        params = await a2a.submit(skill_id=skill_id)
        await a2a.worker.run_task(params)
        assert a2a.agent.run_stream_events.call_args.kwargs["target_action"] == expected

    def test_extracts_user_input(self):
        msg: Message = Message(
            role="user",
            parts=[A2ATextPart(kind="text", text="find active subs")],
            kind="message",
            message_id="m1",
        )
        assert A2AWorker._extract_user_input([msg]) == "find active subs"


class TestA2AWorkerStreaming:
    """Tests for A2AWorker.run_task_streaming() — inline SSE event generation."""

    @pytest.fixture
    def a2a(self):
        storage = InMemoryStorage()
        agent = AsyncMock()
        worker = A2AWorker(agent=agent, broker=InMemoryBroker(), storage=storage)

        async def submit(user_text="show subscriptions", skill_id=None):
            msg: Message = Message(
                role="user",
                parts=[A2ATextPart(kind="text", text=user_text)],
                kind="message",
                message_id=str(uuid.uuid4()),
            )
            context_id = f"ctx-{uuid.uuid4()}"
            task = await storage.submit_task(context_id, msg)
            metadata = {"skill_id": skill_id} if skill_id else {}
            return task, context_id, Message(**{**msg, "metadata": metadata})

        return type("A2AFixture", (), {"worker": worker, "agent": agent, "storage": storage, "submit": submit})

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_stream_yields_working_and_completed(self, _mock_db, a2a):
        a2a.agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

        task, context_id, msg = await a2a.submit()
        chunks = [chunk async for chunk in a2a.worker.run_task_streaming(task["id"], context_id, msg, "req-1")]
        events = parse_sse_events(chunks)

        assert len(events) == 2
        assert events[0]["result"]["kind"] == "status-update"
        assert events[0]["result"]["status"]["state"] == "working"
        assert events[0]["result"]["final"] is False
        assert events[1]["result"]["kind"] == "status-update"
        assert events[1]["result"]["status"]["state"] == "completed"
        assert events[1]["result"]["final"] is True

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_stream_yields_artifact_events(self, _mock_db, a2a):
        a2a.agent.run_stream_events = MagicMock(
            return_value=mock_event_stream(
                make_artifact_event("run_search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )

        task, context_id, msg = await a2a.submit()
        chunks = [chunk async for chunk in a2a.worker.run_task_streaming(task["id"], context_id, msg, "req-1")]
        events = parse_sse_events(chunks)

        kinds = [e["result"]["kind"] for e in events]
        assert kinds == ["status-update", "artifact-update", "status-update"]
        assert events[1]["result"]["artifact"]["parts"][0]["kind"] == "text"

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_stream_yields_failed_on_error(self, _mock_db, a2a):
        async def failing_stream(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # noqa: F401 — makes this an async generator

        a2a.agent.run_stream_events = MagicMock(return_value=failing_stream())

        task, context_id, msg = await a2a.submit()
        chunks = [chunk async for chunk in a2a.worker.run_task_streaming(task["id"], context_id, msg, "req-1")]
        events = parse_sse_events(chunks)

        assert events[-1]["result"]["kind"] == "status-update"
        assert events[-1]["result"]["status"]["state"] == "failed"
        assert events[-1]["result"]["final"] is True

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_send_unchanged(self, _mock_db, a2a):
        """Confirm message/send still works via the broker-based run_task path."""
        a2a.agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

        params = await a2a.storage.submit_task(
            f"ctx-{uuid.uuid4()}",
            Message(
                role="user",
                parts=[A2ATextPart(kind="text", text="hello")],
                kind="message",
                message_id=str(uuid.uuid4()),
            ),
        )
        send_params = TaskSendParams(
            id=params["id"],
            context_id=params["context_id"],
            message=params["history"][0],
        )
        await a2a.worker.run_task(send_params)

        task = await a2a.storage.load_task(params["id"])
        assert task["status"]["state"] == "completed"

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_send_returns_completed_task(self, _mock_db, a2a):
        """run_task_inline returns a task with state=completed and artifacts."""
        a2a.agent.run_stream_events = MagicMock(
            return_value=mock_event_stream(
                make_artifact_event("run_search", SAMPLE_ARTIFACT),
                make_text_result_event("Execution completed"),
            )
        )

        task, context_id, msg = await a2a.submit()
        completed = await a2a.worker.run_task_inline(task["id"], context_id, msg)

        assert completed["status"]["state"] == "completed"
        assert len(completed["artifacts"]) == 1
        assert completed["artifacts"][0]["parts"][0]["kind"] == "text"
        agent_msgs = [m for m in completed["history"] if m["role"] == "agent"]
        assert len(agent_msgs) == 1

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_send_returns_failed_on_error(self, _mock_db, a2a):
        """run_task_inline marks the task as failed on exception."""

        async def failing_stream(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # noqa: F401 — makes this an async generator

        a2a.agent.run_stream_events = MagicMock(return_value=failing_stream())

        task, context_id, msg = await a2a.submit()
        with pytest.raises(RuntimeError, match="boom"):
            await a2a.worker.run_task_inline(task["id"], context_id, msg)

        stored = await a2a.storage.load_task(task["id"])
        assert stored["status"]["state"] == "failed"


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


class TestA2AWorkerStateFallback:
    """Tests for A2A state-based fallback when events don't capture results."""

    @pytest.fixture
    def a2a(self):
        storage = InMemoryStorage()
        agent = AsyncMock()
        worker = A2AWorker(agent=agent, broker=InMemoryBroker(), storage=storage)

        async def submit(user_text="show subscriptions", skill_id=None):
            msg: Message = Message(
                role="user",
                parts=[A2ATextPart(kind="text", text=user_text)],
                kind="message",
                message_id=str(uuid.uuid4()),
            )
            context_id = f"ctx-{uuid.uuid4()}"
            task = await storage.submit_task(context_id, msg)
            metadata = {"skill_id": skill_id} if skill_id else {}
            return TaskSendParams(
                id=task["id"],
                context_id=context_id,
                message={**msg, "metadata": metadata},
            )

        return type("A2AFixture", (), {"worker": worker, "agent": agent, "storage": storage, "submit": submit})

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_inline_state_fallback_over_execution_completed(self, _mock_db, a2a):
        """When agent yields only 'Execution completed' but state has tool steps, use state."""

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

        a2a.agent.run_stream_events = MagicMock(side_effect=stream_with_state_mutation)

        params = await a2a.submit()
        await a2a.worker.run_task(params)

        task = await a2a.storage.load_task(params["id"])
        agent_msgs = [m for m in task["history"] if m["role"] == "agent"]
        assert len(agent_msgs) == 1
        assert "Searched 5 subscriptions" in agent_msgs[0]["parts"][0]["text"]
        assert agent_msgs[0]["parts"][0]["text"] != "Execution completed"


class TestA2AStreamingStateFallback:
    """Tests for streaming state-based fallback."""

    @pytest.fixture
    def a2a(self):
        storage = InMemoryStorage()
        agent = AsyncMock()
        worker = A2AWorker(agent=agent, broker=InMemoryBroker(), storage=storage)

        async def submit(user_text="show subscriptions", skill_id=None):
            msg: Message = Message(
                role="user",
                parts=[A2ATextPart(kind="text", text=user_text)],
                kind="message",
                message_id=str(uuid.uuid4()),
            )
            context_id = f"ctx-{uuid.uuid4()}"
            task = await storage.submit_task(context_id, msg)
            metadata = {"skill_id": skill_id} if skill_id else {}
            return task, context_id, Message(**{**msg, "metadata": metadata})

        return type("A2AFixture", (), {"worker": worker, "agent": agent, "storage": storage, "submit": submit})

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_streaming_state_fallback(self, _mock_db, a2a):
        """Streaming: state fallback used when only 'Execution completed' is yielded."""

        async def stream_with_state_mutation(*args, **kwargs):
            deps = kwargs.get("deps")
            if deps:
                deps.state.memory.start_turn(deps.state.user_input)
                deps.state.memory.start_step("Search")
                deps.state.memory.record_tool_step(ToolStep(step_type="run_search", description="Found 3 workflows"))
                deps.state.memory.finish_step()
                deps.state.memory.complete_turn(assistant_answer="Found 3 workflows.")
            yield make_text_result_event("Execution completed")

        a2a.agent.run_stream_events = MagicMock(side_effect=stream_with_state_mutation)

        task, context_id, msg = await a2a.submit()
        chunks = [chunk async for chunk in a2a.worker.run_task_streaming(task["id"], context_id, msg, "req-1")]
        events = parse_sse_events(chunks)

        completed_event = events[-1]
        assert completed_event["result"]["status"]["state"] == "completed"
        message_text = completed_event["result"]["status"]["message"]["parts"][0]["text"]
        assert "Found 3 workflows" in message_text
        assert message_text != "Execution completed"


class TestA2AEndpoint:
    """HTTP-level tests for StreamingFastA2A._agent_run_endpoint.

    Verifies that message/send and message/stream both return completed results
    (not fire-and-forget "submitted" status) when called via the A2A HTTP API.
    """

    @pytest.fixture
    async def a2a_app(self):
        storage = InMemoryStorage()
        broker = InMemoryBroker()
        agent = AsyncMock()
        worker = A2AWorker(agent=agent, broker=broker, storage=storage)

        @asynccontextmanager  # type: ignore[untyped-decorator]
        async def _noop_lifespan(_app: FastA2A) -> AsyncIterator[None]:
            yield

        app = StreamingFastA2A(
            storage=storage,
            broker=broker,
            worker=worker,
            name="Test Agent",
            url="http://localhost:8000",
            description="Test",
            lifespan=_noop_lifespan,
        )

        async with app.task_manager:
            yield type("A2AAppFixture", (), {"app": app, "agent": agent, "worker": worker, "storage": storage})

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
                    "kind": "message",
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
        task = body["result"]
        assert task["status"]["state"] == "completed"
        assert len(task["artifacts"]) >= 1

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
        assert events[0]["result"]["status"]["state"] == "working"
        assert events[-1]["result"]["status"]["state"] == "completed"
        assert events[-1]["result"]["final"] is True


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
