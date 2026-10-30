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

"""AG-UI adapter — the agent as the backend of a frontend that speaks AG-UI (the chat popup).

pydantic-ai's AG-UI adapter turns a run into the protocol's events: the frontend sends the thread's
messages with each run and streams the text, the tool calls and their results back. Two things are this
module's: tool results that carry an artifact go out as the artifact's lightweight reference (the UI
fetches the full result lazily), and the workflow form-fill skill runs in front of the model, answered by
a person through AG-UI interrupts (``interrupts``).

A form turn works as on the other protocols (``adapters.a2a``, ``adapters.chat``): before the run, the
conversation's open form and the person's response carried by the run (a resume entry) go on the run
state; the skill claims the turn inside the run; after it, the skill's reply is streamed in the
transport's words and its stop ends the run with an interrupt outcome, recorded on the session before the
state is snapshotted. The state the run uses is the server's own, keyed on the thread: what the frontend
sends as AG-UI state is not read.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ag_ui.core import (
    BaseEvent,
    EventType,
    Interrupt,
    RunAgentInput,
    RunFinishedEvent,
    RunFinishedInterruptOutcome,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallResultEvent,
)
from orchestrator.core.db.models import AgentRunTable
from pydantic_ai.messages import FunctionToolResultEvent, TextPart, ToolReturnPart
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.ag_ui import AGUIEventStream as _BaseAGUIEventStream
from pydantic_ai.ui.ag_ui._utils import INTERRUPT_ID_PREFIX
from sqlalchemy.orm import Session
from structlog import get_logger

from orchestrator_agent.adapters.ag_ui.interrupts import AGUIInterrupts, supports_interrupts
from orchestrator_agent.adapters.hitl import HumanInTheLoop
from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.persistence import PostgresStatePersistence, StoredThread, dump_messages, load_messages
from orchestrator_agent.state import Reply, SearchState

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = get_logger(__name__)

# A client's way of showing a stop, over this protocol: the run input in, an interrupt out (on ``RUN_FINISHED``).
AGUIHitl = HumanInTheLoop[RunAgentInput, Interrupt]

FORM_RESPONSE = "[form response]"  # the input of a run that is the person's response to a stop
NOT_OPEN = "That form is no longer open. Ask again to start it."
AGENT_TYPE = "agui"  # how this route's runs are recorded
AGUI_AGENT_TYPES = (AGENT_TYPE, "search")  # the threads that are this route's ("search": before the chats API)


class ThreadForbiddenError(Exception):
    """The thread belongs to someone else, or to another route: this caller may not continue or read it."""


@dataclass
class AGUIEventStream(_BaseAGUIEventStream[Any, Any]):
    """The protocol's event stream, with what this agent adds.

    Tool results whose metadata is a ``ToolArtifact`` are sent as a lightweight JSON reference instead of
    the full payload, so the UI can fetch full results lazily. A form-fill reply left on the run state by
    the skill is streamed in the transport's words rather than as the skill's JSON, and its stop ends the
    run with an interrupt outcome instead of a plain finish.
    """

    hitl: AGUIHitl | None = field(default=None, kw_only=True)
    deps: StateDeps[SearchState] | None = field(default=None, kw_only=True)
    _text_streamed: bool = field(default=False, init=False)

    @property
    def _state(self) -> SearchState | None:
        return self.deps.state if self.deps is not None else None

    def _words(self, reply: Reply) -> str:
        return self.hitl.words(reply) if self.hitl is not None else reply.text

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        result = event.part
        if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
            yield ToolCallResultEvent(
                message_id=self.new_message_id(),
                type=EventType.TOOL_CALL_RESULT,
                role="tool",
                tool_call_id=result.tool_call_id,
                content=result.metadata.model_dump_json(),
            )
            return

        # Default behavior for all other tools
        async for e in super().handle_function_tool_result(event):
            yield e

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[BaseEvent]:
        # The conversation so far, with this run's messages: the thread's memory, replayed on the next run.
        if (state := self._state) is not None:
            state.message_history = dump_messages(event.result.all_messages())
        async for agui_event in super().handle_run_result(event):
            yield agui_event

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseEvent]:
        # The skill's reply is the run's text (the model was skipped): the person reads it in the
        # transport's words; a calling agent would get the JSON.
        state = self._state
        if state is not None and state.form_reply is not None:
            part = TextPart(content=self._words(state.form_reply))
        self._text_streamed = True
        async for event in super().handle_text_start(part, follows_text):
            yield event

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
        state = self._state
        reply = state.form_reply if state is not None else None
        if reply is not None and not self._text_streamed and not self._error and not self._cancelled_run:
            # The skill answered in place of the model, and the run streamed no text for it (a skipped model
            # request has no stream): the person still reads the reply, before the stop.
            message_id = self.new_message_id()
            yield TextMessageStartEvent(message_id=message_id, timestamp=self._get_timestamp())
            yield TextMessageContentEvent(
                message_id=message_id, delta=self._words(reply), timestamp=self._get_timestamp()
            )
            yield TextMessageEndEvent(message_id=message_id, timestamp=self._get_timestamp())
        interrupt = None
        if (
            reply is not None
            and state is not None
            and self.hitl is not None
            and self.run_input is not None
            and not self._error
            and not self._cancelled_run
        ):
            # A stop: the interrupt the client shows, remembered on the session so the response maps back.
            interrupt = self.hitl.pause(reply, state.form_fill, self.run_input)
        if interrupt is None:
            async for event in super().after_stream():
                yield event
            return
        yield RunFinishedEvent(
            thread_id=self.thread_id,
            run_id=self.run_id,
            outcome=RunFinishedInterruptOutcome(interrupts=[interrupt]),
            timestamp=self._get_timestamp(),
        )


@dataclass
class _AGUIAdapter(AGUIAdapter[Any, Any]):
    """AGUIAdapter that uses AGUIEventStream, with the transport and the run state it needs."""

    hitl: AGUIHitl | None = field(default=None, kw_only=True)
    deps: StateDeps[SearchState] | None = field(default=None, kw_only=True)

    def build_event_stream(self) -> AGUIEventStream:
        return AGUIEventStream(
            self.run_input, accept=self.accept, ag_ui_version=self.ag_ui_version, hitl=self.hitl, deps=self.deps
        )


class AGUIWorker:
    """Orchestrates AG-UI request handling: state setup, the form-fill transport, persistence, stream creation."""

    @staticmethod
    async def run_request(
        agent: "WFOAgent",
        run_input: RunAgentInput,
        db_session: Session,
        *,
        auth_token: str | None = None,
        owner: str | None = None,
        transports: Sequence[AGUIHitl] | None = None,
    ) -> AsyncIterator[str]:
        """Execute the full AG-UI lifecycle and return an SSE event iterator.

        ``owner`` is the subject of the person asking (None when the deployment runs without OAuth2): a thread
        belongs to the subject that created it, and only that subject continues it; a thread another route
        wrote is not this route's to continue (``ThreadForbiddenError``). ``transports`` are the clients a form can
        be shown through; the one that claims the request is used (by default AG-UI interrupts, for a client
        whose protocol version has them).
        """
        transports = (AGUIInterrupts(),) if transports is None else transports
        transport = next((t for t in transports if t.shows_stops(run_input)), None)

        run_id = UUID(run_input.run_id)
        thread_id = run_input.thread_id
        persistence = PostgresStatePersistence(thread_id=thread_id, run_id=run_id, session=db_session)

        # The conversation's memory and open form, the server's own. A form being filled spans runs: it is
        # what the skill continues, with the person's response to its stop when this run carries one.
        thread = await persistence.load_thread()
        state = AGUIWorker._owned(thread, owner)

        # Create or get agent run record
        agent_run = db_session.get(AgentRunTable, run_id)
        if not agent_run:
            agent_run = AgentRunTable(run_id=run_id, thread_id=thread_id, agent_type=AGENT_TYPE)
            db_session.add(agent_run)
            db_session.commit()
            logger.debug("Created new agent run", run_id=str(run_id), thread_id=thread_id)

        state.user_input = FORM_RESPONSE if run_input.resume else AGUIWorker._extract_user_input(run_input)
        state.run_id = run_id
        logger.debug("Run state", user_input=state.user_input[:100], resumed=bool(run_input.resume))

        response = transport.read(state.form_fill, run_input) if transport is not None else None
        if isinstance(response, Interrupt):  # the transport answers this run itself: the stop, shown again
            return AGUIWorker._shows(run_input, response)
        if run_input.resume and response is None:
            # A response to a stop that is not pending (the form closed): say so, and run nothing.
            return AGUIWorker._says(run_input, NOT_OPEN)
        state.hitl = transport is not None
        state.form_input = response
        state.form_reply = None
        deps = StateDeps(state)

        adapter = _AGUIAdapter(
            agent=agent,
            run_input=AGUIWorker._prepare_run_input(run_input, state),
            ag_ui_version=AGUIWorker._version(run_input) or AGUIAdapter.ag_ui_version,
            hitl=transport,
            deps=deps,
        )

        # The thread's earlier turns are the run's history: the client sends only this run's messages.
        history = load_messages(state.message_history)

        async def _stream_with_persistence() -> AsyncIterator[str]:
            """Open the MCP session, stream AG-UI events, persist, and commit on completion."""
            try:
                with bind_outbound_token(auth_token):
                    async with agent:
                        event_stream = adapter.encode_stream(adapter.run_stream(deps=deps, message_history=history))
                        async for event_str in event_stream:
                            yield event_str
                # The state the run used (pydantic-ai rebuilds it from the run input); a stop is on it.
                await persistence.snapshot(deps.state)
                db_session.commit()
            except Exception as e:
                logger.error("Error in agent stream", error=str(e), exc_info=True)
                db_session.rollback()
                raise

        return _stream_with_persistence()

    @staticmethod
    def _owned(thread: StoredThread | None, owner: str | None) -> SearchState:
        """The thread's state for this caller: a new thread, or one of theirs; an unowned one of this route becomes theirs."""
        if thread is None:
            return SearchState(owner=owner)
        if thread.agent_type not in AGUI_AGENT_TYPES:
            raise ThreadForbiddenError(f"thread {thread.thread_id} belongs to the {thread.agent_type} route")
        if thread.state.owner is not None and thread.state.owner != owner:
            raise ThreadForbiddenError(f"thread {thread.thread_id} belongs to someone else")
        thread.state.owner = owner
        return thread.state

    @staticmethod
    def _version(run_input: RunAgentInput) -> str | None:
        """The AG-UI version the events are written for: the client's, when it says which it speaks; else the default."""
        version = run_input.protocol_version
        return version if version and supports_interrupts(version) else None

    @staticmethod
    def _prepare_run_input(run_input: RunAgentInput, state: SearchState) -> RunAgentInput:
        """The run input as pydantic-ai gets it: the server's state in place of the client's, and only its own resume entries.

        pydantic-ai builds the run's state from the input's state. It also maps every resume entry back to a
        tool call of its own (its approval interrupts carry a prefix it knows), so the entry that answered
        the form-fill skill's stop, already read into the state, is left out.
        """
        resume = [entry for entry in run_input.resume or [] if entry.interrupt_id.startswith(INTERRUPT_ID_PREFIX)]
        return run_input.model_copy(update={"state": AGUIWorker._raw_state(state), "resume": resume or None})

    @staticmethod
    def _raw_state(state: SearchState) -> dict[str, Any]:
        """The state as the run input carries it, with the fields of this turn that a snapshot leaves out."""
        raw = state.model_dump(mode="json")
        raw["hitl"] = state.hitl
        raw["form_input"] = dataclasses.asdict(state.form_input) if state.form_input is not None else None
        raw["form_reply"] = None
        return raw

    @staticmethod
    def _extract_user_input(run_input: RunAgentInput) -> str:
        """Extract the most recent user message from RunAgentInput messages."""
        for msg in reversed(run_input.messages):
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content
        return ""

    @staticmethod
    async def _says(run_input: RunAgentInput, text: str) -> AsyncIterator[str]:
        """A run that is one message from the agent, without the agent: the events of it, encoded."""
        message_id = uuid.uuid4().hex
        finished = RunFinishedEvent(
            thread_id=run_input.thread_id, run_id=run_input.run_id, outcome=RunFinishedSuccessOutcome()
        )
        async for encoded in AGUIWorker._events(
            run_input,
            TextMessageStartEvent(message_id=message_id),
            TextMessageContentEvent(message_id=message_id, delta=text),
            TextMessageEndEvent(message_id=message_id),
            finished,
        ):
            yield encoded

    @staticmethod
    async def _shows(run_input: RunAgentInput, interrupt: Interrupt) -> AsyncIterator[str]:
        """A run that only shows a stop, without the agent: the events of it, encoded."""
        outcome = RunFinishedInterruptOutcome(interrupts=[interrupt])
        finished = RunFinishedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id, outcome=outcome)
        async for encoded in AGUIWorker._events(run_input, finished):
            yield encoded

    @staticmethod
    async def _events(run_input: RunAgentInput, *events: BaseEvent) -> AsyncIterator[str]:
        """The run's start, then ``events``, encoded for the client's AG-UI version."""
        version = AGUIWorker._version(run_input) or AGUIEventStream.ag_ui_version
        stream = AGUIEventStream(run_input, ag_ui_version=version)
        yield stream.encode_event(RunStartedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id))
        for event in events:
            yield stream.encode_event(event)


__all__ = [
    "AGENT_TYPE",
    "AGUI_AGENT_TYPES",
    "FORM_RESPONSE",
    "NOT_OPEN",
    "AGUIEventStream",
    "AGUIHitl",
    "AGUIWorker",
    "ThreadForbiddenError",
]
