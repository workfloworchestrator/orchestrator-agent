"""Mock event factories for adapter output tests.

These factories produce the same event types that `agent.run_stream_events()`
yields, so adapter tests can exercise transformation logic without any LLM or
DB calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from itertools import repeat, takewhile
from typing import Any, AsyncIterator

# Set required env vars before any orchestrator_agent imports trigger Settings instantiation.
os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from ag_ui.core import RunAgentInput, UserMessage
from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

from orchestrator_agent.artifacts import ToolArtifact

pytestmark = pytest.mark.search


def make_artifact_event(
    tool_name: str,
    artifact: ToolArtifact,
    content: Any | None = None,
    tool_call_id: str = "tc_1",
) -> FunctionToolResultEvent:
    """FunctionToolResultEvent whose ToolReturnPart carries a ToolArtifact.

    Content defaults to the artifact's JSON representation — matching what
    real tools return via ToolReturn(metadata=..., content=artifact.model_dump_json()).
    """
    if content is None:
        content = artifact.model_dump_json()
    part = ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id,
        metadata=artifact,
    )
    return FunctionToolResultEvent(part=part)


def make_non_artifact_event(
    tool_name: str,
    content: Any = "plain result",
    tool_call_id: str = "tc_2",
) -> FunctionToolResultEvent:
    """FunctionToolResultEvent without ToolArtifact metadata."""
    part = ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id,
    )
    return FunctionToolResultEvent(part=part)


def make_text_result_event(text: str = "Execution completed") -> AgentRunResultEvent[str]:
    """AgentRunResultEvent with a text output."""
    return AgentRunResultEvent(result=AgentRunResult(output=text))


def minimal_run_input() -> RunAgentInput:
    """Minimal AG-UI RunAgentInput for testing."""
    return RunAgentInput(
        thread_id="t1",
        run_id="r1",
        state={},
        messages=[UserMessage(id="m1", role="user", content="show subs")],
        tools=[],
        context=[],
        forwarded_props={},
    )


async def mock_event_stream(*events: Any) -> AsyncIterator[Any]:
    """Async generator that yields the given events in order."""
    for event in events:
        yield event


def parse_sse_events(raw_chunks: list[bytes]) -> list[dict]:
    """Parse SSE event chunks into JSON objects."""
    events = []
    for chunk in raw_chunks:
        text = chunk.decode()
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def spin(stop: threading.Event, limit: float = 3.0) -> None:
    """Pure-Python CPU work until ``stop`` is set (or ``limit`` seconds pass): what a heavy request does.

    On the event loop it blocks every other request; the limit only keeps a failing test from hanging.
    """
    deadline = time.perf_counter() + limit
    deque(takewhile(lambda _: not stop.is_set() and time.perf_counter() < deadline, repeat(None)), maxlen=0)


async def loop_lags(ticks: int = 5, every: float = 0.01) -> list[float]:
    """How long the event loop took to wake up from each of ``ticks`` sleeps, beyond the sleep itself.

    A loop that is blocked cannot wake the sleeper: the lag is the length of the block. Run it while the heavy
    work under test is going on and assert on ``max``.
    """

    async def tick() -> float:
        started = time.perf_counter()
        await asyncio.sleep(every)
        return time.perf_counter() - started - every

    return [await tick() for _ in range(ticks)]


# What a request may be held up by another one's heavy work and still count as served promptly. A blocked loop
# is held up for the whole of the work (the ``spin`` limit), so this is generous, not a timing assertion.
RESPONSIVE = 1.0
