"""What the form-carrying adapters share: one turn at a time per conversation."""

from __future__ import annotations

import asyncio
import csv
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart, ToolReturnPart
from structlog.testing import capture_logs

from orchestrator_agent.artifacts import QueryArtifact, RenderedBlock
from orchestrator_agent.rendering.tables import aggregate_to_markdown
from orchestrator_agent.scratch import SavedFile, file_line
from orchestrator_agent.turn import ConversationLocks, _run_model, _ToolRounds
from tests.conftest import make_artifact_event, make_non_artifact_event, make_text_result_event, mock_event_stream


class TestConversationLocks:
    @pytest.mark.asyncio
    async def test_turns_of_one_conversation_run_one_at_a_time(self):
        locks, running, overlapped = ConversationLocks(), 0, False

        async def turn(conversation: str) -> None:
            nonlocal running, overlapped
            async with locks.turn(conversation):
                running += 1
                overlapped = overlapped or running > 1
                await asyncio.sleep(0)
                running -= 1

        await asyncio.gather(*(turn("chat-1") for _ in range(5)))
        assert not overlapped
        assert not locks  # nothing is kept once no turn runs or waits

    @pytest.mark.asyncio
    async def test_other_conversations_are_not_held_up(self):
        locks = ConversationLocks()
        async with locks.turn("chat-1"):
            assert len(locks) == 1
            await asyncio.wait_for(self._one_turn(locks, "chat-2"), timeout=1)
            assert len(locks) == 1  # chat-2's turn is over, chat-1's still runs
        assert not locks

    @staticmethod
    async def _one_turn(locks: ConversationLocks, conversation: str) -> None:
        async with locks.turn(conversation):
            pass

    @pytest.mark.asyncio
    async def test_a_failing_turn_releases_its_conversation(self):
        locks = ConversationLocks()
        with pytest.raises(RuntimeError):
            async with locks.turn("chat-1"):
                raise RuntimeError("the run failed")
        assert not locks


def _block_event(text: str, *, total_results: int = 3) -> FunctionToolResultEvent:
    """A query result carrying a rendered table, as the aggregate and search tools return it."""
    artifact = QueryArtifact(
        description="aggregate",
        query_id="q",
        total_results=total_results,
        rendered_block=RenderedBlock(type="markdown", content=text),
    )
    return make_artifact_event("aggregate", artifact)


def _round_events(*results: str | None) -> list[Any]:
    """One tool round: a call per result, then the results; ``None`` is a scalar result without a block.

    A string is a rendered block; ``"~text"`` is a block of a query that found nothing.
    """
    calls = [FunctionToolCallEvent(part=ToolCallPart(tool_name="aggregate", args={})) for _ in results]
    outcomes = [
        make_non_artifact_event("count", content="42")
        if text is None
        else _block_event(text.removeprefix("~"), total_results=0 if text.startswith("~") else 3)
        for text in results
    ]
    return [*calls, *outcomes]


def _agent_streaming(*events: Any) -> MagicMock:
    agent = MagicMock()

    @asynccontextmanager
    async def _run(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield mock_event_stream(*events, make_text_result_event("The answer."))

    agent.run_stream_events = MagicMock(side_effect=_run)
    return agent


class TestRunModelBlocks:
    """Only the last round that produced a block is appended; ``on_artifact`` still sees every artifact."""

    @pytest.mark.parametrize(
        ("rounds", "appended"),
        [
            pytest.param([["probe"], ["final"]], ["final"], id="probe-round-then-final-round"),
            pytest.param(
                [["probe"], ["final-1", "final-2"]], ["final-1", "final-2"], id="final-round-keeps-all-its-blocks"
            ),
            pytest.param([["probe-1", "probe-2"], ["final"]], ["final"], id="parallel-probes-dropped"),
            pytest.param([["probe"], [None]], ["probe"], id="final-scalar-round-keeps-previous-blocks"),
            pytest.param([["probe"], ["final"], [None]], ["final"], id="trailing-scalar-round-keeps-last-block-round"),
            pytest.param([["~empty"], ["final"]], ["final"], id="empty-result-skipped"),
            pytest.param([["probe"], ["~empty"]], ["probe"], id="empty-final-round-keeps-previous-blocks"),
            pytest.param([["~empty"]], [], id="only-empty-result-appends-nothing"),
            pytest.param([["final", "final"]], ["final"], id="same-block-twice-appended-once"),
            pytest.param([["a", "b", "a"]], ["a", "b"], id="duplicates-dropped-order-kept"),
            pytest.param([[None]], [], id="no-block-appends-nothing"),
            pytest.param([], [], id="no-tool-calls"),
        ],
    )
    @pytest.mark.asyncio
    async def test_appends_the_blocks_of_the_last_round_that_has_one(
        self, rounds: list[list[str | None]], appended: list[str]
    ) -> None:
        events = [event for results in rounds for event in _round_events(*results)]

        output = await _run_model(_agent_streaming(*events), "q", MagicMock(), None, None)

        assert output == "\n\n".join(["The answer.", *appended])

    @pytest.mark.parametrize(
        "rounds",
        [
            pytest.param([["probe"], ["final"]], id="probe-and-final"),
            pytest.param([["~empty"], ["a", "a"], [None]], id="empty-duplicate-and-scalar"),
        ],
    )
    @pytest.mark.asyncio
    async def test_on_artifact_fires_for_every_artifact(self, rounds: list[list[str | None]]) -> None:
        events = [event for results in rounds for event in _round_events(*results)]
        seen: list[str] = []

        async def on_artifact(part: ToolReturnPart) -> None:
            seen.append(part.tool_name)

        await _run_model(_agent_streaming(*events), "q", MagicMock(), None, on_artifact)

        expected = sum(text is not None for results in rounds for text in results)
        assert seen == ["aggregate"] * expected


CONVERSATION = "2d2900fa-e77a-428c-a385-7f46cf16a457"


def _buckets(count: int, query_id: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "results": [{"group_values": {"customer": f"c{n}"}, "aggregations": {"count": n}} for n in range(count)],
    }


def _call() -> FunctionToolCallEvent:
    return FunctionToolCallEvent(part=ToolCallPart(tool_name="aggregate", args={}))


def _rows_event(payload: dict[str, Any], *, rendered: bool = True) -> FunctionToolResultEvent:
    """An aggregate result as the agent receives it: the payload is the content, the table is the block."""
    markdown = aggregate_to_markdown(payload)
    artifact = QueryArtifact(
        description="aggregate",
        query_id=payload["query_id"],
        total_results=len(payload["results"]),
        rendered_block=RenderedBlock(type="markdown", content=markdown) if rendered and markdown else None,
    )
    return make_artifact_event("aggregate", artifact, content=payload)


def _data_rows(table: str) -> list[str]:
    """The bucket rows of a table: the buckets are named c0, c1, ..."""
    return [line for line in table.splitlines() if re.match(r"\| c\d+ \|", line)]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestRunModelScratch:
    """The final round is saved for the calling agent, and the reply says so; otherwise nothing changes."""

    @pytest.mark.asyncio
    async def test_final_round_is_saved_and_the_reply_names_it(self, tmp_path: Path) -> None:
        events = [_call(), _rows_event(_buckets(150, "q9"))]

        output = await _run_model(_agent_streaming(*events), "q", MagicMock(), None, None, scratch=tmp_path)

        assert len(_csv_rows(tmp_path / "q-q9.csv")) == 150
        assert output.endswith(file_line(tmp_path.name, "/data", SavedFile("q-q9.csv", 150, ("customer", "count"))))
        assert len(_data_rows(output)) == 20
        assert "| **Total** |" in output
        assert "… 150 rows in total; all rows are in /data/q-q9.csv" in output

    @pytest.mark.asyncio
    async def test_probe_rounds_are_not_saved(self, tmp_path: Path) -> None:
        events = [_call(), _rows_event(_buckets(3, "q1")), _call(), _rows_event(_buckets(4, "q2"))]

        await _run_model(_agent_streaming(*events), "q", MagicMock(), None, None, scratch=tmp_path)

        assert os.listdir(tmp_path) == ["q-q2.csv"]

    @pytest.mark.asyncio
    async def test_small_result_is_saved_and_keeps_its_whole_table(self, tmp_path: Path) -> None:
        payload = _buckets(3, "q1")

        output = await _run_model(
            _agent_streaming(_call(), _rows_event(payload)), "q", MagicMock(), None, None, scratch=tmp_path
        )

        saved = SavedFile("q-q1.csv", 3, ("customer", "count"))
        assert output == "\n\n".join(
            ["The answer.", aggregate_to_markdown(payload) or "", file_line(tmp_path.name, "/data", saved)]
        )

    @pytest.mark.asyncio
    async def test_a_saved_result_without_a_block_still_gets_its_file_line(self, tmp_path: Path) -> None:
        events = [_call(), _call(), _rows_event(_buckets(3, "qa")), _rows_event(_buckets(2, "qb"), rendered=False)]

        output = await _run_model(_agent_streaming(*events), "q", MagicMock(), None, None, scratch=tmp_path)

        assert sorted(os.listdir(tmp_path)) == ["q-qa.csv", "q-qb.csv"]
        assert output.endswith(file_line(tmp_path.name, "/data", SavedFile("q-qb.csv", 2, ("customer", "count"))))
        assert file_line(tmp_path.name, "/data", SavedFile("q-qa.csv", 3, ("customer", "count"))) in output

    @pytest.mark.asyncio
    async def test_a_result_over_the_file_limit_is_reported_and_keeps_its_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("orchestrator_agent.turn.agent_settings.SCRATCH_MAX_FILE_BYTES", 10)
        payload = _buckets(150, "q9")

        output = await _run_model(
            _agent_streaming(_call(), _rows_event(payload)), "q", MagicMock(), None, None, scratch=tmp_path
        )

        assert os.listdir(tmp_path) == []
        assert output == "\n\n".join(
            [
                "The answer.",
                aggregate_to_markdown(payload) or "",
                "Not saved for run_python: 150 rows exceed the scratch file limit",
            ]
        )

    @pytest.mark.asyncio
    async def test_no_scratch_is_unchanged(self) -> None:
        payload = _buckets(150, "q9")

        output = await _run_model(
            _agent_streaming(_call(), _rows_event(payload)), "q", MagicMock(), None, None, scratch=None
        )

        assert output == "\n\n".join(["The answer.", aggregate_to_markdown(payload) or ""])

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    async def test_unwritable_scratch_leaves_the_answer_unchanged(self, tmp_path: Path) -> None:
        folder = tmp_path / CONVERSATION
        folder.mkdir()
        events = [_call(), _rows_event(_buckets(150, "q9"))]
        expected = await _run_model(_agent_streaming(*events), "q", MagicMock(), None, None, scratch=None)

        folder.chmod(0o500)
        try:
            with capture_logs() as logs:
                output = await _run_model(_agent_streaming(*events), "q", MagicMock(), None, None, scratch=folder)
        finally:
            folder.chmod(0o700)

        assert output == expected
        assert os.listdir(folder) == []
        (error,) = [log for log in logs if log["log_level"] == "error"]
        assert (error["conversation"], error["tool_name"]) == (CONVERSATION, "aggregate")
        assert error["error"]


def _label(part: ToolReturnPart) -> str:
    """The text of the block of a query result; the content of any other result."""
    match part:
        case ToolReturnPart(metadata=QueryArtifact(rendered_block=RenderedBlock(content=text))):
            return text
        case _:
            return str(part.content)


class TestToolRounds:
    @pytest.mark.parametrize(
        ("rounds", "expected"),
        [
            pytest.param([["probe"], ["final"]], ["final"], id="the-final-round-not-the-probe"),
            pytest.param([["probe"], [None]], ["probe"], id="a-later-round-without-a-block-is-skipped"),
            pytest.param([["a", None, "b"]], ["a", "42", "b"], id="results-without-a-block-stay-in-their-round"),
            pytest.param([[None]], [], id="no-block-no-results"),
        ],
    )
    def test_final_results_are_the_parts_of_the_round_that_final_blocks_takes(
        self, rounds: list[list[str | None]], expected: list[str]
    ) -> None:
        tool_rounds = _ToolRounds()
        for event in (event for results in rounds for event in _round_events(*results)):
            match event:
                case FunctionToolCallEvent():
                    tool_rounds.call()
                case FunctionToolResultEvent(part=part):
                    tool_rounds.result(part)

        assert [_label(part) for part in tool_rounds.final_results()] == expected
