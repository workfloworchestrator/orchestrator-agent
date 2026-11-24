# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One heavy request does not hold up another request, or the agent card the readiness probe reads.

Two requests go through the A2A app over ASGI, in one process on one event loop, as they do in a pod. The heavy
one is forced to do CPU-bound work at a boundary where a large result is processed: the work holds on (a spin)
until the test has seen the others served, so the test does not depend on how fast the machine is. When the work
blocks the loop, the others wait for the spin's limit and the test fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import httpx  # noqa: E402
import pytest  # noqa: E402
from a2a.helpers import new_data_part  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart  # noqa: E402

from orchestrator_agent.adapters.a2a import A2AAdapter  # noqa: E402
from orchestrator_agent.artifacts import QueryArtifact  # noqa: E402
from tests.conftest import RESPONSIVE, make_text_result_event, mock_event_stream, spin  # noqa: E402

HEAVY_QUESTION, CHEAP_QUESTION = "aggregate every port", "how many products?"
A2A_HEADERS = {"A2A-Version": "1.0"}


def _jsonrpc(question: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {"message": {"role": "ROLE_USER", "parts": [{"text": question}], "messageId": str(uuid.uuid4())}},
    }


def _agent() -> MagicMock:
    """An agent that answers the heavy question with a query result, and any other with its answer only."""
    agent = MagicMock()

    @asynccontextmanager
    async def run_stream_events(question: str, **kwargs: Any) -> AsyncIterator[Any]:
        artifact = QueryArtifact(description="aggregate", query_id="q", total_results=1)
        query_result = ToolReturnPart(
            "aggregate", {"results": [{"group_values": {"a": "b"}}]}, "call-1", metadata=artifact
        )
        heavy = [FunctionToolResultEvent(part=query_result)] if question == HEAVY_QUESTION else []
        yield mock_event_stream(*heavy, make_text_result_event(f"Answer to: {question}"))

    agent.run_stream_events = MagicMock(side_effect=run_stream_events)
    return agent


def _first_call_spins(real: Callable[..., Any], started: threading.Event, stop: threading.Event) -> Callable[..., Any]:
    """``real``, except that the first call (the heavy request's) does CPU-bound work until ``stop`` is set."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not started.is_set():
            started.set()
            spin(stop)
        return real(*args, **kwargs)

    return wrapped


@pytest.fixture
def app() -> Iterator[FastAPI]:
    with patch("orchestrator_agent.adapters.a2a.adapter.PostgresStatePersistence") as persistence:
        persistence.return_value.load_state = AsyncMock(return_value=None)
        persistence.return_value.snapshot = AsyncMock()
        with patch("orchestrator.core.db.db"):
            adapter = A2AAdapter(_agent(), url="http://localhost:8080/")
            app = FastAPI()
            adapter.add_routes(app)
            yield app


@pytest.mark.parametrize(
    "heavy_function",
    [
        pytest.param("orchestrator_agent.adapters.a2a.adapter.new_data_part", id="large-result-published-as-artifact"),
        pytest.param("orchestrator_agent.turn.dump_messages", id="large-history-stored-after-the-run"),
    ],
)
async def test_a_heavy_request_does_not_hold_up_another_request_or_the_agent_card(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, heavy_function: str
):
    module, name = heavy_function.rsplit(".", 1)
    real = getattr(__import__(module, fromlist=[name]), name)
    started, stop = threading.Event(), threading.Event()
    monkeypatch.setattr(heavy_function, _first_call_spins(real, started, stop))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        began = time.perf_counter()
        heavy = asyncio.create_task(client.post("/", json=_jsonrpc(HEAVY_QUESTION), headers=A2A_HEADERS))
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)

        cheap, card = await asyncio.gather(
            client.post("/", json=_jsonrpc(CHEAP_QUESTION), headers=A2A_HEADERS),
            client.get("/.well-known/agent-card.json"),
        )
        served_in = time.perf_counter() - began
        heavy_was_going_on = not heavy.done()
        stop.set()
        heavy_response = await heavy

    assert served_in < RESPONSIVE, "the others waited for the heavy request's CPU work"
    assert heavy_was_going_on, "the others were served only after the heavy request was over"
    assert card.status_code == 200 and card.json()["name"] == "WFO Agent"
    cheap_task = cheap.json()["result"]["task"]
    assert cheap_task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert cheap_task["status"]["message"]["parts"][0]["text"] == f"Answer to: {CHEAP_QUESTION}"
    heavy_task = heavy_response.json()["result"]["task"]
    assert heavy_task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert json.dumps(heavy_task["artifacts"])  # the heavy request still delivered its result


async def test_the_agent_card_answers_in_time_while_a_heavy_request_runs(app: FastAPI, monkeypatch: pytest.MonkeyPatch):
    """The readiness probe reads the agent card with a timeout: it must not wait for a heavy request."""
    started, stop = threading.Event(), threading.Event()
    monkeypatch.setattr(
        "orchestrator_agent.adapters.a2a.adapter.new_data_part", _first_call_spins(new_data_part, started, stop)
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        began = time.perf_counter()
        heavy = asyncio.create_task(client.post("/", json=_jsonrpc(HEAVY_QUESTION), headers=A2A_HEADERS))
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
        responses = await asyncio.gather(*(client.get("/.well-known/agent-card.json") for _ in range(5)))
        answered_in = time.perf_counter() - began
        stop.set()
        await heavy

    assert [response.status_code for response in responses] == [200] * 5
    assert answered_in < RESPONSIVE
