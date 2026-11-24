"""What the form-carrying adapters share: one turn at a time per conversation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart, ToolReturnPart

from orchestrator_agent.artifacts import QueryArtifact, RenderedBlock
from orchestrator_agent.turn import ConversationLocks, _run_model
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
