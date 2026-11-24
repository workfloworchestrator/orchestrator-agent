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

"""One turn of a conversation, as the protocol adapters that carry workflow forms run it.

A2A and chat completions differ in how a request arrives and how the answer leaves. What happens in
between is the same, and is here: one turn at a time per conversation (``ConversationLocks``), and the
agent run on the conversation's memory and open form, with the human's response as the form-fill
skill's input (``run_turn``). Loading the state, reading the response out of the request, recording a
stop and snapshotting stay with the adapter: they are its transport's, or must happen in its order.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from orchestrator.core.db.models import AgentRunTable
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResultEvent
from sqlalchemy.orm import Session

from orchestrator_agent.agent import new_deps
from orchestrator_agent.artifacts import QueryArtifact, RenderedBlock, ToolArtifact
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.offload import off_loop
from orchestrator_agent.persistence import dump_messages, load_messages
from orchestrator_agent.scratch import SavedFile, Skipped, cut_down, file_line, not_saved_line, save_result
from orchestrator_agent.settings import agent_settings
from orchestrator_agent.state import FormInput, SearchState

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)

NO_RESULTS = "No results"  # what an adapter answers when a run produced no text

# Called with each tool result that carries an artifact, as it happens (a transport that can deliver one).
OnArtifact = Callable[[ToolReturnPart], Awaitable[None]]


class ConversationLocks:
    """One turn at a time per conversation.

    The state is loaded, changed and snapshotted per request, and a caller may send two messages for one
    conversation at once. A lock lives only while a turn of its conversation runs or waits.
    """

    def __init__(self) -> None:
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._users: Counter[str] = Counter()

    def __len__(self) -> int:
        """The conversations with a turn running or waiting."""
        return len(self._users)

    @asynccontextmanager
    async def turn(self, conversation: str) -> AsyncGenerator[None, None]:
        self._users[conversation] += 1
        try:
            async with self._locks[conversation]:
                yield
        finally:
            self._users[conversation] -= 1
            if not self._users[conversation]:
                del self._users[conversation]
                self._locks.pop(conversation, None)


async def run_turn(
    agent: "WFOAgent",
    db_session: Session,
    *,
    thread: str,
    run_id: UUID,
    agent_type: str,
    prior_state: SearchState | None,
    user_input: str,
    hitl: bool,
    form_input: FormInput | None,
    auth_token: str | None,
    on_artifact: OnArtifact | None = None,
    scratch: Path | None = None,
) -> tuple[SearchState, str]:
    """Run the agent for one turn of ``thread``; the run's state and its answer.

    The run is recorded, the conversation's earlier turns are replayed as its message history, and its
    open form is continued with ``form_input`` (the human's response to the pending stop, if the request
    carried one). ``hitl`` says the caller can show a person a stop: a form needs it. The answer is the
    agent's prose plus the deterministically rendered chart/table blocks; a form-fill reply is on the
    state (``form_reply``). The caller snapshots the state, after recording a stop on it.

    ``scratch`` is the conversation's scratch folder (see :mod:`orchestrator_agent.scratch`; its name is the
    conversation's id): the rows of the final tool round are saved there, and the answer names the files.
    """
    deps = new_deps(user_input=user_input)
    deps.state.run_id = run_id
    db_session.add(AgentRunTable(run_id=run_id, thread_id=thread, agent_type=agent_type))
    db_session.commit()

    deps.state.hitl = hitl
    deps.state.form_fill = prior_state.form_fill if prior_state is not None else None
    deps.state.form_input = form_input
    # Multi-turn memory: the caller only needs to send the latest user turn.
    message_history = await off_loop(load_messages, prior_state.message_history if prior_state else [])

    with bind_outbound_token(auth_token):
        output = await _run_model(agent, user_input, deps, message_history, on_artifact, scratch)
    return deps.state, output


@dataclass
class _ToolRounds:
    """The tool results of a run, grouped by tool round, each with its rendered block.

    A round is the tool results produced between two model responses; the model sees one round before it
    decides on the next, so a probe (a schema check, an aggregation over every location) and the query
    it prepared are different rounds. The event stream has no explicit round marker, but a tool call that
    follows a tool result can only come from a new model response, which is where a round starts.

    A result is kept with its block: the block of a non-empty query result, ``None`` for any other result.
    """

    rounds: list[list[tuple[ToolReturnPart, str | None]]] = field(default_factory=lambda: [[]])
    _has_results: bool = False

    def call(self) -> None:
        """A tool call was made: after results it is the first of a new round."""
        if self._has_results:
            self.rounds.append([])
            self._has_results = False

    def result(self, part: object) -> None:
        """A tool result arrived; a non-empty query result brings its block to the current round."""
        self._has_results = True
        match part:
            case ToolReturnPart(metadata=QueryArtifact(total_results=n, rendered_block=RenderedBlock() as block)) if n:
                self.rounds[-1].append((part, block.to_markdown()))
            case ToolReturnPart():
                self.rounds[-1].append((part, None))

    def final_round(self) -> list[tuple[ToolReturnPart, str | None]]:
        """The results, each with its block, of the last round that has a block; empty when none has."""
        return next((r for r in reversed(self.rounds) if any(block is not None for _, block in r)), [])

    def final_blocks(self) -> list[str]:
        """The blocks of the last round that has any, each once, in order."""
        return list(dict.fromkeys(block for _, block in self.final_round() if block is not None))

    def final_results(self) -> list[ToolReturnPart]:
        """The tool results of the same round ``final_blocks`` takes its blocks from."""
        return [part for part, _ in self.final_round()]


async def _run_model(
    agent: "WFOAgent",
    user_input: str,
    deps: Any,
    message_history: list[ModelMessage] | None,
    on_artifact: OnArtifact | None,
    scratch: Path | None = None,
) -> str:
    """Stream one model run: artifacts go out as they happen; the prose plus rendered blocks is returned.

    With a ``scratch`` folder, the rows of the results behind those blocks are saved there as well, and the
    blocks are followed by what the calling agent needs to read them.
    """
    final_output = ""
    tool_rounds = _ToolRounds()
    async with agent.run_stream_events(user_input, deps=deps, message_history=message_history) as events:
        async for event in events:
            match event:
                case FunctionToolCallEvent():
                    tool_rounds.call()
                case FunctionToolResultEvent(part=ToolReturnPart(metadata=ToolArtifact()) as result):
                    if on_artifact is not None:
                        await on_artifact(result)
                    tool_rounds.result(result)
                case FunctionToolResultEvent(part=result):
                    tool_rounds.result(result)
                case AgentRunResultEvent():
                    final_output = str(event.result.output)
                    deps.state.message_history = await off_loop(dump_messages, event.result.all_messages())
                case PartDeltaEvent():
                    pass
                case _:
                    logger.debug("Agent run: event", event_type=type(event).__name__)
    # Append chart/table blocks after the agent's prose. The agent only sees raw data (no ToolReturn.content
    # relay), so it just summarises — the adapter guarantees the block appears. Only the non-empty blocks of
    # the last round that produced any are appended: earlier rounds are the model probing (a schema, a
    # grouping over every location) on its way to the real query, and their tables would bury the answer.
    # ``on_artifact`` has already seen every artifact, so a client that renders them itself loses nothing.
    if blocks := await _reply_blocks(tool_rounds, scratch):
        final_output = final_output.rstrip() + "\n\n" + "\n\n".join(blocks)
    return final_output


async def _reply_blocks(tool_rounds: _ToolRounds, scratch: Path | None) -> list[str]:
    """What follows the agent's prose: the blocks of the final round, and where their rows were saved.

    A result that was saved has its table cut down when it is long and is followed by the line that names its
    file, or by the line that says it was too large. A result that could not be saved is shown as it is.
    """
    final = tool_rounds.final_round()
    saves = [await _save(scratch, part) for part, _ in final]
    conversation = scratch.name if scratch is not None else ""
    shown = dict.fromkeys((block, saved) for (_, block), saved in zip(final, saves, strict=True))
    return [segment for block, saved in shown for segment in _segments(block, saved, conversation)]


def _segments(block: str | None, saved: SavedFile | Skipped | None, conversation: str) -> list[str]:
    """The reply text of one result: its block (cut down when its rows are saved) and the line about its file."""
    shown = [] if block is None else [block]
    virtual_dir = agent_settings.SCRATCH_VIRTUAL_DIR
    match saved:
        case SavedFile():
            max_rows, preview_rows = (
                agent_settings.SCRATCH_REPLY_TABLE_MAX_ROWS,
                agent_settings.SCRATCH_REPLY_PREVIEW_ROWS,
            )
            return [
                *(cut_down(text, saved, virtual_dir, max_rows, preview_rows) for text in shown),
                file_line(conversation, virtual_dir, saved),
            ]
        case Skipped():
            return [*shown, not_saved_line(saved)]
        case _:
            return shown


async def _save(scratch: Path | None, part: ToolReturnPart) -> SavedFile | Skipped | None:
    """Save the rows of ``part`` in ``scratch``; ``None`` when there is nothing to save or it failed (logged).

    Saving is best-effort: whatever goes wrong, the answer is the one there would be without a scratch folder.
    """
    if scratch is None:
        return None
    try:
        return await off_loop(save_result, scratch, part.tool_name, part.content, agent_settings.SCRATCH_MAX_FILE_BYTES)
    except Exception as exc:  # noqa: BLE001 - a failed save must never change or fail the answer
        logger.error("Cannot save the tool result", conversation=scratch.name, tool_name=part.tool_name, error=str(exc))
        return None


__all__ = ["NO_RESULTS", "ConversationLocks", "OnArtifact", "run_turn"]
