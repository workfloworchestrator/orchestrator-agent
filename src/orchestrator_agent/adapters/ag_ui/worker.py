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
messages with each run and streams the text, the tool calls and their results back. It also maps the
framework's tool approval onto AG-UI interrupts, which is how the start of a workflow is approved: the run
stops on ``start_workflow``, the client shows the interrupt, and its answer comes back as that call's
result. Three things are this module's: tool results that carry an artifact go out as the artifact's
lightweight reference (the UI fetches the full result lazily); a form goes out as an activity message that
names its tool calls (``activity``); and the answer a run carries is matched to the start that is waiting
(``interrupts``).

The state the run uses is the server's own, keyed on the thread: what the frontend sends as AG-UI state is
not read.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    EventType,
    RunAgentInput,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallResultEvent,
)
from orchestrator.core.db.models import AgentRunTable
from pydantic_ai.messages import FunctionToolResultEvent, ToolCallPart, ToolReturnPart
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.ag_ui import AGUIEventStream as _BaseAGUIEventStream
from sqlalchemy.orm import Session
from structlog import get_logger

from orchestrator_agent.adapters.ag_ui.activity import (
    FORM_ACTIVITY,
    FORM_TOOLS,
    closed_form,
    form_activity,
    form_calls,
    form_groups,
)
from orchestrator_agent.adapters.ag_ui.contract import carried
from orchestrator_agent.adapters.ag_ui.interrupts import (
    answers,
    denial,
    pending_start,
    rejected,
    supports_interrupts,
    words,
)
from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.persistence import PostgresStatePersistence, StoredThread, dump_messages, load_messages
from orchestrator_agent.state import SearchState

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = get_logger(__name__)

NOT_OPEN = "That form is no longer open. Ask again to start it."
# What the model is told, as the result of the start that was waiting, when the person did not approve it.
REJECTED = "The person rejected the start. The form is closed and nothing was started."
NOT_APPROVED = "Not approved: nothing was started. The person wants to talk about it first and wrote:"
AGENT_TYPE = "agui"  # how this route's runs are recorded
AGUI_AGENT_TYPES = (AGENT_TYPE, "search")  # the threads that are this route's ("search": before the chats API)


class ThreadForbiddenError(Exception):
    """The thread belongs to someone else, or to another route: this caller may not continue or read it."""


class EmptyRunError(Exception):
    """The request carries neither a message nor an answer to a waiting start: there is nothing to run."""


class ApprovalPendingError(Exception):
    """A start waits for the person's decision, and the request does not carry one: decide first."""


@dataclass
class AGUIEventStream(_BaseAGUIEventStream[Any, Any]):
    """The protocol's event stream, with what this agent adds.

    Tool results whose metadata is a ``ToolArtifact`` are sent as a lightweight JSON reference instead of
    the full payload, so the UI can fetch full results lazily. For a client that is shown approvals, a
    form goes out as one activity message that names the tool calls it is filled with (``activity``).
    """

    deps: StateDeps[SearchState] | None = field(default=None, kw_only=True)
    closed: tuple[str, str] | None = field(default=None, kw_only=True)  # a form the request itself closed
    _form_calls: list[ToolCallPart] = field(default_factory=list, init=False)  # the thread's form tool calls so far
    _form: tuple[str, str] | None = field(default=None, init=False)  # the form this run touched: its id and workflow

    def __post_init__(self) -> None:
        super().__post_init__()
        self._form = self.closed
        if (state := self._state) is not None and state.hitl:
            self._form_calls = form_calls(load_messages(state.message_history) or [])
        self._see_form()

    @property
    def _state(self) -> SearchState | None:
        return self.deps.state if self.deps is not None else None

    def _see_form(self) -> None:
        state = self._state
        if state is not None and state.hitl and state.form_fill is not None:
            self._form = (state.form_fill.id, state.form_fill.workflow_key)

    def _form_snapshot(self) -> list[BaseEvent]:
        """The form this run touched, as its activity message stands now; nothing when the run touched none."""
        self._see_form()
        state = self._state
        if self._form is None or state is None:
            return []
        message_id, workflow_key = self._form
        calls = (form_groups(self._form_calls) or [[]])[-1]
        open_form = state.form_fill if state.form_fill is not None and state.form_fill.id == message_id else None
        activity = form_activity(open_form, calls) if open_form else closed_form(calls, workflow_key)
        return [
            ActivitySnapshotEvent(
                message_id=message_id,
                activity_type=FORM_ACTIVITY,
                content=activity.model_dump(mode="json"),
                replace=True,
                timestamp=self._get_timestamp(),
            )
        ]

    def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[BaseEvent]:  # type: ignore[override]
        if (state := self._state) is not None and state.hitl and part.tool_name in FORM_TOOLS:
            self._form_calls.append(part)
        return super().handle_tool_call_start(part)

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        result = event.part
        if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
            yield ToolCallResultEvent(
                message_id=self.new_message_id(),
                type=EventType.TOOL_CALL_RESULT,
                role="tool",
                tool_call_id=result.tool_call_id,
                content=carried(result.metadata),
            )
            return

        # Default behavior for all other tools
        async for e in super().handle_function_tool_result(event):
            yield e
        if result.tool_name in FORM_TOOLS:
            for snapshot in self._form_snapshot():  # the form moved on: its message is replaced
                yield snapshot

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[BaseEvent]:
        # The conversation so far, with this run's messages: the thread's memory, replayed on the next run.
        if (state := self._state) is not None:
            state.message_history = dump_messages(event.result.all_messages())
        async for agui_event in super().handle_run_result(event):
            yield agui_event

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
        if not self._error and not self._cancelled_run:
            for snapshot in self._form_snapshot():  # as the run leaves the form: waiting for approval, or closed
                yield snapshot
        async for event in super().after_stream():  # an approval the run stopped on goes out as its interrupt
            yield event


@dataclass
class _AGUIAdapter(AGUIAdapter[Any, Any]):
    """AGUIAdapter that uses AGUIEventStream, with the run state it needs."""

    deps: StateDeps[SearchState] | None = field(default=None, kw_only=True)
    closed: tuple[str, str] | None = field(default=None, kw_only=True)

    def build_event_stream(self) -> AGUIEventStream:
        return AGUIEventStream(
            self.run_input, accept=self.accept, ag_ui_version=self.ag_ui_version, deps=self.deps, closed=self.closed
        )


class AGUIWorker:
    """Orchestrates AG-UI request handling: state setup, the answer to a waiting start, persistence, stream creation."""

    @staticmethod
    async def run_request(
        agent: "WFOAgent",
        run_input: RunAgentInput,
        db_session: Session,
        *,
        auth_token: str | None = None,
        owner: str | None = None,
    ) -> AsyncIterator[str]:
        """Execute the full AG-UI lifecycle and return an SSE event iterator.

        ``owner`` is the subject of the person asking (None when the deployment runs without OAuth2): a thread
        belongs to the subject that created it, and only that subject continues it; a thread another route
        wrote is not this route's to continue (``ThreadForbiddenError``).
        """
        said = AGUIWorker._extract_user_input(run_input)
        if not said and not run_input.resume:
            raise EmptyRunError("The request carries neither a message nor an answer to a waiting approval.")
        run_id = UUID(run_input.run_id)
        thread_id = run_input.thread_id
        persistence = PostgresStatePersistence(thread_id=thread_id, run_id=run_id, session=db_session)

        thread = await persistence.load_thread()
        state = AGUIWorker._owned(thread, owner)

        agent_run = db_session.get(AgentRunTable, run_id)
        if not agent_run:
            agent_run = AgentRunTable(run_id=run_id, thread_id=thread_id, agent_type=AGENT_TYPE)
            db_session.add(agent_run)
            db_session.commit()
            logger.debug("Created new agent run", run_id=str(run_id), thread_id=thread_id)

        history = load_messages(state.message_history)
        waiting = pending_start(history or [])
        if run_input.resume and waiting is None and not said:
            # An answer to a start that is not waiting (the form closed), and nothing else: say so, run nothing.
            return AGUIWorker._says(run_input, NOT_OPEN)
        state.user_input = said
        state.run_id = run_id
        # A client that speaks an AG-UI with interrupts can be shown an approval: a start needs one.
        state.hitl = supports_interrupts(run_input.protocol_version)
        logger.debug("Run state", user_input=said[:100], resumed=bool(run_input.resume))
        entries = answers(waiting, run_input.resume)
        closed = None
        if waiting is not None and not entries:
            # A conversation cannot go on past a start that waits: the client has to send the decision.
            raise ApprovalPendingError(
                f"A workflow start waits for approval (interrupt int-{waiting.tool_call_id}): answer it first."
            )
        if waiting is not None and rejected(entries[0]) and (session := state.form_fill) is not None:
            # The person rejected the start outright: the form is over, and the model is told so.
            closed = (session.id, session.workflow_key)
            fs.close(session)
            state.form_fill = None
            entries = [denial(waiting, REJECTED)]
        elif waiting is not None and (wrote := words(entries[0])) is not None:
            # They did not approve and wrote something: the model is told whose words those are.
            entries = [denial(waiting, f"{NOT_APPROVED} {wrote!r}")]
        deps = StateDeps(state)

        # pydantic-ai builds the run's state from the input's state, and turns each resume entry into the
        # result of the tool call it answers: only the answer to the waiting start is passed on.
        prepared = run_input.model_copy(update={"state": AGUIWorker._raw_state(state), "resume": entries or None})
        adapter = _AGUIAdapter(
            agent=agent,
            run_input=prepared,
            ag_ui_version=AGUIWorker._version(run_input) or AGUIAdapter.ag_ui_version,
            deps=deps,
            closed=closed,
        )

        async def _stream_with_persistence() -> AsyncIterator[str]:
            """Stream AG-UI events, persist, and commit on completion."""
            try:
                with bind_outbound_token(auth_token):
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
        """The thread's state for this caller: a new thread, or one that is theirs."""
        if thread is None:
            return SearchState(owner=owner)
        if thread.agent_type not in AGUI_AGENT_TYPES:
            raise ThreadForbiddenError(f"thread {thread.thread_id} belongs to the {thread.agent_type} route")
        if thread.state.owner != owner:
            raise ThreadForbiddenError(f"thread {thread.thread_id} is not the caller's")
        return thread.state

    @staticmethod
    def _version(run_input: RunAgentInput) -> str | None:
        """The AG-UI version the events are written for: the client's, when it says which it speaks; else the default."""
        version = run_input.protocol_version
        return version if version and supports_interrupts(version) else None

    @staticmethod
    def _raw_state(state: SearchState) -> dict[str, Any]:
        """The state as the run input carries it, with the field of this turn that a snapshot leaves out."""
        return {**state.model_dump(mode="json"), "hitl": state.hitl}

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
    "NOT_OPEN",
    "AGUIEventStream",
    "AGUIWorker",
    "ApprovalPendingError",
    "EmptyRunError",
    "ThreadForbiddenError",
]
