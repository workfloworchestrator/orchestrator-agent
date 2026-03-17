"""Adapter output tests — verify each protocol adapter transforms agent events correctly.

We mock `agent.run_stream_events()` to yield pre-built event sequences.
No LLM calls, no DB calls — just adapter transformation logic.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ag_ui.core import ToolCallResultEvent
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Message, TaskSendParams
from fasta2a.schema import TextPart as A2ATextPart
from fasta2a.storage import InMemoryStorage

from orchestrator_agent.adapters.a2a import A2AWorker
from orchestrator_agent.adapters.ag_ui import AGUIEventStream
from orchestrator_agent.adapters.mcp import MCPWorker
from orchestrator_agent.adapters.stream import NO_RESULTS, collect_stream_output
from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.events import AgentStepActiveEvent
from orchestrator_agent.state import TaskAction

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

    @staticmethod
    def _parse_sse_events(raw_chunks: list[bytes]) -> list[dict]:
        events = []
        for chunk in raw_chunks:
            text = chunk.decode()
            for line in text.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    @pytest.mark.asyncio
    @patch("orchestrator.db.db")
    async def test_message_stream_yields_working_and_completed(self, _mock_db, a2a):
        a2a.agent.run_stream_events = MagicMock(return_value=mock_event_stream(make_text_result_event("Done")))

        task, context_id, msg = await a2a.submit()
        chunks = [chunk async for chunk in a2a.worker.run_task_streaming(task["id"], context_id, msg, "req-1")]
        events = self._parse_sse_events(chunks)

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
        events = self._parse_sse_events(chunks)

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
        events = self._parse_sse_events(chunks)

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
