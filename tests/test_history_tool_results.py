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

"""A tool result in a thread's stored history is bounded, so a huge one cannot make every later call run out of memory."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.run import AgentRunResultEvent  # noqa: E402
from structlog.testing import capture_logs  # noqa: E402

from orchestrator_agent.artifacts import DataArtifact, QueryArtifact, RenderedBlock  # noqa: E402
from orchestrator_agent.persistence import (  # noqa: E402
    MAX_HISTORY_TOOL_METADATA_CHARS,
    MAX_HISTORY_TOOL_RESULT_CHARS,
    dump_messages,
    load_messages,
    truncate_tool_results,
)
from orchestrator_agent.turn import _run_model  # noqa: E402
from tests.conftest import mock_event_stream  # noqa: E402

LIMIT = MAX_HISTORY_TOOL_RESULT_CHARS
ROWS = [{"group": f"sub-{i}", "count": i} for i in range(1645)]  # the aggregate over every subscription
HUGE_PAYLOAD = {"aggregations": ROWS}
HUGE_JSON = json.dumps(HUGE_PAYLOAD)
HUGE_SIZE = len(ToolReturnPart("t", HUGE_PAYLOAD).model_response_str())  # as the model reads the payload


def _history(content: Any, *, metadata: Any = None, tool_call_id: str = "call-1") -> list[ModelMessage]:
    """One earlier run: the person asks, the model calls ``aggregate``, the tool answers with ``content``."""
    return [
        ModelRequest(parts=[UserPromptPart(content="count per subscription")]),
        ModelResponse(parts=[ToolCallPart(tool_name="aggregate", args={}, tool_call_id=tool_call_id)]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="aggregate", content=content, tool_call_id=tool_call_id, metadata=metadata)]
        ),
        ModelResponse(parts=[TextPart(content="There are 1645.")]),
    ]


def _tool_return(messages: list[ModelMessage]) -> ToolReturnPart:
    return next(part for message in messages for part in message.parts if isinstance(part, ToolReturnPart))


def _small_artifact() -> DataArtifact:
    return DataArtifact(description="a subscription", entity_id="s1", entity_type="subscription")


def _large_artifact() -> QueryArtifact:
    return QueryArtifact(
        description="aggregate",
        query_id="q",
        total_results=len(ROWS),
        rendered_block=RenderedBlock(type="markdown", content=HUGE_JSON),
    )


def _its_reference() -> QueryArtifact:
    """``_large_artifact`` without the rows it rendered: the query it came from."""
    return QueryArtifact(description="aggregate", query_id="q", total_results=len(ROWS))


def _large_data_artifact() -> DataArtifact:
    return DataArtifact(
        description="x" * (MAX_HISTORY_TOOL_METADATA_CHARS + 1), entity_id="s1", entity_type="subscription"
    )


class TestTruncateToolResults:
    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(HUGE_PAYLOAD, id="json-payload"),
            pytest.param(HUGE_JSON, id="json-string"),
            pytest.param("x" * (LIMIT + 1), id="one-char-over-the-limit"),
            pytest.param(ROWS, id="list-of-rows"),
        ],
    )
    def test_a_large_result_is_cut_to_the_limit(self, content: Any) -> None:
        part = _tool_return(truncate_tool_results(_history(content)))

        assert isinstance(part.content, str)
        assert len(part.content) <= LIMIT

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("a short answer", id="short-string"),
            pytest.param({"count": 42}, id="small-payload"),
            pytest.param("x" * LIMIT, id="exactly-the-limit"),
            pytest.param(None, id="no-content"),
        ],
    )
    def test_a_small_result_is_left_as_it_is(self, content: Any) -> None:
        messages = _history(content)

        assert truncate_tool_results(messages) == messages

    def test_the_start_of_the_result_is_kept_and_the_marker_states_the_original_size(self) -> None:
        part = _tool_return(truncate_tool_results(_history(HUGE_PAYLOAD)))

        text = part.model_response_str()
        assert text.startswith('{"aggregations":[{"group":"sub-0","count":0},')
        assert f"[truncated from {HUGE_SIZE} chars when stored in the conversation history" in text
        assert text.endswith("re-run the query to see all rows]")

    @pytest.mark.parametrize("content", [pytest.param(HUGE_PAYLOAD, id="large"), pytest.param("small", id="small")])
    def test_the_message_structure_is_kept(self, content: Any) -> None:
        messages = _history(content, tool_call_id="call-42")

        truncated = truncate_tool_results(messages)

        assert [type(m) for m in truncated] == [type(m) for m in messages]
        assert [[type(p) for p in m.parts] for m in truncated] == [[type(p) for p in m.parts] for m in messages]
        part = _tool_return(truncated)
        assert (part.tool_call_id, part.tool_name) == ("call-42", "aggregate")
        assert truncated[:2] == messages[:2] and truncated[3] == messages[3]

    def test_the_messages_it_is_given_are_not_changed(self) -> None:
        messages = _history(HUGE_PAYLOAD, metadata=_large_artifact())

        truncate_tool_results(messages)

        part = _tool_return(messages)
        assert part.content == HUGE_PAYLOAD and isinstance(part.metadata, QueryArtifact)

    @pytest.mark.parametrize(
        ("content", "metadata", "expected"),
        [
            pytest.param(HUGE_PAYLOAD, _small_artifact(), _small_artifact(), id="large-result-small-metadata-kept"),
            pytest.param("small", _small_artifact(), _small_artifact(), id="small-result-small-metadata-kept"),
            pytest.param(HUGE_PAYLOAD, _large_artifact(), _its_reference(), id="large-result-query-artifact-stripped"),
            pytest.param("small", _large_artifact(), _its_reference(), id="small-result-query-artifact-stripped"),
            pytest.param(
                "small",
                _large_artifact().model_dump(mode="json"),
                _its_reference().model_dump(mode="json"),
                id="query-artifact-as-stored-json-stripped",
            ),
            pytest.param("small", _large_data_artifact(), None, id="other-large-artifact-dropped"),
            pytest.param("small", {"rows": ROWS}, None, id="large-plain-metadata-dropped"),
            pytest.param("small", None, None, id="no-metadata"),
        ],
    )
    def test_metadata_is_kept_when_small_and_a_query_artifact_keeps_its_reference(
        self, content: Any, metadata: Any, expected: Any
    ) -> None:
        part = _tool_return(truncate_tool_results(_history(content, metadata=metadata)))

        assert part.metadata == expected

    def test_large_metadata_is_over_the_metadata_limit(self) -> None:
        assert len(_large_artifact().model_dump_json()) > MAX_HISTORY_TOOL_METADATA_CHARS
        assert len(_large_data_artifact().model_dump_json()) > MAX_HISTORY_TOOL_METADATA_CHARS
        assert len(_small_artifact().model_dump_json()) < MAX_HISTORY_TOOL_METADATA_CHARS
        assert len(_its_reference().model_dump_json()) < MAX_HISTORY_TOOL_METADATA_CHARS

    @pytest.mark.parametrize(
        "content",
        [pytest.param(HUGE_PAYLOAD, id="large"), pytest.param({"count": 1}, id="small")],
    )
    def test_truncating_twice_changes_nothing(self, content: Any) -> None:
        once = truncate_tool_results(_history(content, metadata=_large_artifact()))

        assert truncate_tool_results(once) == once

    def test_every_truncated_result_is_logged_once_with_its_tool_and_size(self) -> None:
        messages = _history(HUGE_PAYLOAD) + _history("small", tool_call_id="call-2")

        with capture_logs() as logs:
            once = truncate_tool_results(messages)
            truncate_tool_results(once)  # already bounded: nothing more to report

        assert [(log["tool_name"], log["original_chars"], log["log_level"]) for log in logs] == [
            ("aggregate", HUGE_SIZE, "info")
        ]


def _stored_unbounded(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """The history as it was stored before results were bounded."""
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")


class TestStoredHistory:
    def test_a_large_result_is_truncated_when_saved(self) -> None:
        stored = dump_messages(_history(HUGE_PAYLOAD, metadata=_large_artifact()))

        raw = json.dumps(stored)
        assert len(raw) < 2 * LIMIT
        part = stored[2]["parts"][0]
        assert len(part["content"]) <= LIMIT and part["metadata"] == _its_reference().model_dump(mode="json")
        assert part["tool_call_id"] == "call-1"

    def test_a_result_already_stored_whole_is_truncated_when_loaded(self) -> None:
        # What production holds today: the raw history, written before results were bounded.
        stored = _stored_unbounded(_history(HUGE_PAYLOAD, metadata=_large_artifact()))
        assert len(json.dumps(stored)) > 2 * LIMIT

        loaded = load_messages(stored)

        assert loaded is not None
        part = _tool_return(loaded)
        assert len(part.model_response_str()) <= LIMIT and part.metadata == _its_reference().model_dump(mode="json")
        assert part.tool_call_id == "call-1"
        assert [type(m) for m in loaded] == [ModelRequest, ModelResponse, ModelRequest, ModelResponse]

    def test_nothing_stored_is_nothing_to_replay(self) -> None:
        assert load_messages([]) is None

    def test_a_small_history_survives_a_round_trip(self) -> None:
        messages = _history({"count": 42}, metadata=None)

        assert load_messages(dump_messages(messages)) == messages


class TestTheRunSeesItsOwnResultWhole:
    async def test_only_earlier_runs_results_are_bounded(self) -> None:
        """Within a run the model gets the full result it just asked for; the stored history is what is bounded."""
        seen: list[list[ModelMessage]] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(list(messages))
            returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            if len(returned) < 2:  # the earlier run's result is one; this run's has not come yet
                return ModelResponse(parts=[ToolCallPart(tool_name="aggregate", args={}, tool_call_id="call-new")])
            return ModelResponse(parts=[TextPart(content="done")])

        agent = Agent(FunctionModel(model))

        @agent.tool_plain
        def aggregate() -> str:
            return HUGE_JSON

        earlier = load_messages(dump_messages(_history(HUGE_PAYLOAD)))
        result = await agent.run("again", message_history=earlier)

        last_call = [p for m in seen[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
        by_id = {p.tool_call_id: p.content for p in last_call}
        assert by_id["call-new"] == HUGE_JSON  # this run's result, whole
        assert len(str(by_id["call-1"])) <= LIMIT  # the earlier run's, bounded
        new_part = next(p for p in _all_returns(result.all_messages()) if p.tool_call_id == "call-new")
        assert new_part.content == HUGE_JSON  # saving does not reach back into the finished run
        assert max(len(p["content"]) for p in _stored_returns(result.all_messages())) <= LIMIT

    async def test_the_turn_stores_the_bounded_history(self) -> None:
        deps = MagicMock()
        result = MagicMock(output="done")
        result.all_messages.return_value = _history(HUGE_PAYLOAD)
        agent = MagicMock()

        @asynccontextmanager
        async def _run(*args: Any, **kwargs: Any):
            yield mock_event_stream(AgentRunResultEvent(result=result))

        agent.run_stream_events = MagicMock(side_effect=_run)

        await _run_model(agent, "q", deps, None, None)

        stored = deps.state.message_history
        assert len(json.dumps(stored)) < 2 * LIMIT
        assert stored[2]["parts"][0]["tool_call_id"] == "call-1"


def _all_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def _stored_returns(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return [
        part for message in dump_messages(messages) for part in message["parts"] if part["part_kind"] == "tool-return"
    ]
