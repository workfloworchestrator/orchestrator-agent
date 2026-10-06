"""What the form-carrying adapters share: one turn at a time per conversation."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator_agent.turn import ConversationLocks


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
