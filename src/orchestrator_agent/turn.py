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
) -> tuple[SearchState, str]:
    """Run the agent for one turn of ``thread``; the run's state and its answer.

    The run is recorded, the conversation's earlier turns are replayed as its message history, and its
    open form is continued with ``form_input`` (the human's response to the pending stop, if the request
    carried one). ``hitl`` says the caller can show a person a stop: a form needs it. The answer is the
    agent's prose plus the deterministically rendered chart/table blocks; a form-fill reply is on the
    state (``form_reply``). The caller snapshots the state, after recording a stop on it.
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
        output = await _run_model(agent, user_input, deps, message_history, on_artifact)
    return deps.state, output


@dataclass
class _ToolRounds:
    """The rendered blocks of a run's tool results, grouped by tool round.

    A round is the tool results produced between two model responses; the model sees one round before it
    decides on the next, so a probe (a schema check, an aggregation over every location) and the query
    it prepared are different rounds. The event stream has no explicit round marker, but a tool call that
    follows a tool result can only come from a new model response, which is where a round starts.
    """

    rounds: list[list[str]] = field(default_factory=lambda: [[]])
    _has_results: bool = False

    def call(self) -> None:
        """A tool call was made: after results it is the first of a new round."""
        if self._has_results:
            self.rounds.append([])
            self._has_results = False

    def result(self, part: object) -> None:
        """A tool result arrived; a non-empty query result adds its block to the current round."""
        self._has_results = True
        match part:
            case ToolReturnPart(metadata=QueryArtifact(total_results=n, rendered_block=RenderedBlock() as block)) if n:
                self.rounds[-1].append(block.to_markdown())

    def final_blocks(self) -> list[str]:
        """The blocks of the last round that has any, each once, in order."""
        return next((list(dict.fromkeys(blocks)) for blocks in reversed(self.rounds) if blocks), [])


async def _run_model(
    agent: "WFOAgent",
    user_input: str,
    deps: Any,
    message_history: list[ModelMessage] | None,
    on_artifact: OnArtifact | None,
) -> str:
    """Stream one model run: artifacts go out as they happen; the prose plus rendered blocks is returned."""
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
    if blocks := tool_rounds.final_blocks():
        final_output = final_output.rstrip() + "\n\n" + "\n\n".join(blocks)
    return final_output


__all__ = ["NO_RESULTS", "ConversationLocks", "OnArtifact", "run_turn"]
