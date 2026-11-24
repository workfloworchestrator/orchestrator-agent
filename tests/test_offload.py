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

"""Heavy work on a large result runs off the event loop, so it cannot hold up every other request.

``off_loop`` runs a function on a worker thread. Each boundary where a large result is processed (artifact
rendering, the history dump and load, the state snapshot, the A2A artifact part, the MCP endpoint's data) is
tested to call its heavy function through it: the function is replaced by CPU work that holds on until the test
releases it, and the loop must keep waking up meanwhile. Run on the loop itself, the same work blocks the probe
for its whole duration, which ``test_the_probe_sees_a_blocked_loop`` shows.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest  # noqa: E402
from a2a.helpers import new_data_part  # noqa: E402
from ag_ui.core import RunAgentInput, UserMessage  # noqa: E402
from pydantic_ai import ModelRetry  # noqa: E402
from pydantic_ai.messages import ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent  # noqa: E402
from pydantic_ai.tools import ToolDefinition  # noqa: E402
from pydantic_ai.ui import StateDeps  # noqa: E402

from orchestrator_agent.adapters.a2a import WFOAgentExecutor  # noqa: E402
from orchestrator_agent.adapters.ag_ui import AGUIEventStream, AGUIWorker  # noqa: E402
from orchestrator_agent.adapters.mcp import MCPWorker  # noqa: E402
from orchestrator_agent.capabilities.behavior import PluginCapability  # noqa: E402
from orchestrator_agent.capabilities.spec import PluginSpec  # noqa: E402
from orchestrator_agent.mcp_client import build_core_toolset  # noqa: E402
from orchestrator_agent.offload import off_loop  # noqa: E402
from orchestrator_agent.persistence import PostgresStatePersistence  # noqa: E402
from orchestrator_agent.state import SearchState  # noqa: E402
from orchestrator_agent.tool_names import SEARCH_TOOL  # noqa: E402
from orchestrator_agent.turn import _run_model, run_turn  # noqa: E402
from tests.conftest import (  # noqa: E402
    RESPONSIVE,
    loop_lags,
    make_text_result_event,
    minimal_run_input,
    mock_event_stream,
    spin,
)

# A scenario runs the real code path of one boundary, with the heavy function it calls replaced by ``heavy``.
Scenario = Callable[[pytest.MonkeyPatch, Callable[[], Any]], Coroutine[Any, Any, None]]


class TestOffLoop:
    async def test_runs_the_function_on_another_thread_and_returns_its_value(self):
        thread = await off_loop(threading.get_ident)

        assert thread != threading.get_ident()
        assert await off_loop(lambda a, b=0: a + b, 1, b=2) == 3

    async def test_raises_what_the_function_raises(self):
        def fail() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await off_loop(fail)

    async def test_the_function_sees_the_context_of_the_caller(self):
        outbound: contextvars.ContextVar[str] = contextvars.ContextVar("outbound", default="unset")
        outbound.set("the person's token")

        assert await off_loop(outbound.get) == "the person's token"

    @pytest.mark.parametrize(
        ("offloaded", "blocked"),
        [
            pytest.param(False, True, id="on-the-loop-blocks-it"),
            pytest.param(True, False, id="off-the-loop-leaves-it-free"),
        ],
    )
    async def test_the_probe_sees_a_blocked_loop(self, offloaded: bool, blocked: bool):
        """The same CPU work on the loop holds the probe up for its whole length; off the loop it does not."""
        stop = threading.Event()
        threading.Timer(0.4, stop.set).start()  # the work lasts 0.4 s

        probe = asyncio.create_task(loop_lags())
        await asyncio.sleep(0)  # the probe takes its first tick before the work starts
        if offloaded:
            await off_loop(spin, stop)
        else:
            spin(stop)
        lags = await probe

        assert (max(lags) > 0.3) is blocked


def _agent_that_answers() -> MagicMock:
    """An agent whose run streams one event: the answer. ``run_stream_events`` is an async context manager."""
    agent = MagicMock()

    @asynccontextmanager
    async def run(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield mock_event_stream(make_text_result_event("Done"))

    agent.run_stream_events = MagicMock(side_effect=run)
    return agent


async def _history_dump(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.turn.dump_messages", lambda messages: heavy() or [])
    agent = _agent_that_answers()
    await _run_model(agent, "q", MagicMock(), None, None)


async def _history_load(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.turn.load_messages", lambda stored: heavy() or None)
    agent = _agent_that_answers()
    await run_turn(
        agent,
        MagicMock(),
        thread="t",
        run_id=uuid.uuid4(),
        agent_type="a2a",
        prior_state=SearchState(message_history=[{"kind": "request", "parts": []}]),
        user_input="q",
        hitl=False,
        form_input=None,
        auth_token=None,
    )


async def _artifact_builder(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    spec = PluginSpec(id="search", description="d", instructions="i", defer_loading=False, tools=["SEARCH_TOOL"])
    capability = PluginCapability(spec, lambda tool, args, payload: heavy())
    await capability.after_tool_execute(
        cast("Any", SimpleNamespace()),
        call=ToolCallPart(tool_name=SEARCH_TOOL, args={}),
        tool_def=ToolDefinition(name=SEARCH_TOOL, parameters_json_schema={"properties": {}}),
        args={},
        result={"results": []},
    )


async def _a2a_artifact_part(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr(
        "orchestrator_agent.adapters.a2a.adapter.new_data_part", lambda data: heavy() or new_data_part(data)
    )
    updater = MagicMock()
    updater.add_artifact = AsyncMock()
    await WFOAgentExecutor._publisher(updater)(ToolReturnPart("aggregate", {"results": []}, "call-1"))


async def _state_snapshot(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr(
        "orchestrator_agent.persistence.snapshot_data", lambda state, sequence: heavy() or {"state": {}, "sequence": 0}
    )
    await PostgresStatePersistence("t", uuid.uuid4(), MagicMock()).snapshot(SearchState())


async def _state_read(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.persistence.read_state", lambda data, thread_id: heavy() or SearchState())
    session = MagicMock()
    stored = SimpleNamespace(snapshot_data={}, sequence_number=3, created_at=datetime.now(UTC))
    session.execute.return_value.one_or_none.return_value = (stored, "a2a")
    await PostgresStatePersistence("t", uuid.uuid4(), session).load_thread()


async def _core_rejection(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.mcp_client.explain_core_rejection", lambda error, args: heavy() or "told")
    rejected = AsyncMock(side_effect=ModelRetry("HTTP error 422: rejected"))
    with pytest.raises(ModelRetry, match="told"):
        await cast("Any", build_core_toolset()).process_tool_call(None, rejected, "search", {})


async def _mcp_endpoint_data(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.adapters.mcp._result_with_data", lambda result: heavy() or "answer")
    agent = MagicMock()
    agent.run = AsyncMock(return_value=SimpleNamespace(output="answer"))
    with patch("orchestrator_agent.adapters.mcp.db"):
        await MCPWorker(agent).run("how many?")


async def _agui_history_dump(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.adapters.ag_ui.worker.dump_messages", lambda messages: heavy() or [])
    stream = AGUIEventStream(run_input=minimal_run_input(), deps=StateDeps(SearchState()))
    _ = [event async for event in stream.handle_run_result(AgentRunResultEvent(result=AgentRunResult(output="Done")))]


async def _agui_history_load(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    monkeypatch.setattr("orchestrator_agent.adapters.ag_ui.worker.load_messages", lambda stored: heavy() or None)
    run_input = RunAgentInput(
        thread_id="t",
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="m1", role="user", content="hello")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    session = MagicMock()
    with patch("orchestrator_agent.adapters.ag_ui.worker.PostgresStatePersistence") as persistence:
        persistence.return_value.load_thread = AsyncMock(return_value=None)
        with patch("orchestrator_agent.adapters.ag_ui.worker._AGUIAdapter"):
            await AGUIWorker.run_request(MagicMock(), run_input, session)


def _output_schema_session(schema: dict[str, Any] | None) -> Any:
    """An MCP client session that knows one tool, ``aggregate``, with this output schema (or none)."""
    import anyio
    from mcp import ClientSession

    write, read = anyio.create_memory_object_stream[Any](1)
    session = ClientSession(read, write)
    session._tool_output_schemas = {"aggregate": schema}
    return session


def _structured(content: dict[str, Any] | None) -> Any:
    from mcp.types import CallToolResult

    return CallToolResult(content=[], structuredContent=content)


async def _mcp_result_validation(monkeypatch: pytest.MonkeyPatch, heavy: Callable[[], Any]) -> None:
    import orchestrator_agent.mcp_client  # noqa: F401  (patches the SDK's validation)

    monkeypatch.setattr("jsonschema.validate", lambda content, schema, **kwargs: heavy())
    session = _output_schema_session({"type": "object"})
    await session._validate_tool_result("aggregate", _structured({"results": []}))


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_artifact_builder, id="artifact-table-and-chart-rendering"),
        pytest.param(_history_dump, id="history-dump-a2a-and-chat"),
        pytest.param(_history_load, id="history-load-a2a-and-chat"),
        pytest.param(_a2a_artifact_part, id="a2a-artifact-part"),
        pytest.param(_state_snapshot, id="state-snapshot"),
        pytest.param(_state_read, id="state-read"),
        pytest.param(_mcp_result_validation, id="mcp-result-validation"),
        pytest.param(_core_rejection, id="core-rejection-explanation"),
        pytest.param(_mcp_endpoint_data, id="mcp-endpoint-result-data"),
        pytest.param(_agui_history_dump, id="agui-history-dump"),
        pytest.param(_agui_history_load, id="agui-history-load"),
    ],
)
async def test_heavy_work_at_a_boundary_leaves_the_event_loop_free(scenario: Scenario, monkeypatch: pytest.MonkeyPatch):
    stop, started = threading.Event(), threading.Event()

    def heavy() -> None:
        started.set()
        spin(stop)

    probe = asyncio.create_task(loop_lags(ticks=30))
    run = asyncio.create_task(scenario(monkeypatch, heavy))
    await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
    await asyncio.sleep(0.1)  # the heavy work is going on, and the loop is still turning
    still_working = not run.done()
    stop.set()
    await run
    lags = await probe

    assert started.is_set() and still_working
    assert max(lags) < RESPONSIVE


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"total_results": {"type": "integer"}},
    "required": ["total_results"],
}


@pytest.mark.parametrize(
    ("schema", "content", "invalid"),
    [
        pytest.param(OUTPUT_SCHEMA, {"total_results": 3}, False, id="valid-content-passes"),
        pytest.param(OUTPUT_SCHEMA, {"total_results": "three"}, True, id="invalid-content-is-rejected-as-before"),
        pytest.param(None, {"anything": 1}, False, id="a-tool-without-an-output-schema-is-not-checked"),
    ],
)
async def test_the_sdks_check_of_a_tool_result_still_rejects_what_does_not_match_its_schema(
    schema: dict[str, Any] | None, content: dict[str, Any], invalid: bool
):
    import orchestrator_agent.mcp_client  # noqa: F401  (patches the SDK's validation)

    check = _output_schema_session(schema)._validate_tool_result("aggregate", _structured(content))

    if invalid:
        with pytest.raises(RuntimeError, match="Invalid structured content returned by tool aggregate"):
            await check
    else:
        await check
