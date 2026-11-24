# Copyright 2019-2025 SURF, GÉANT.
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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from orchestrator.core.db.models import AgentRunTable, GraphSnapshotTable
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ToolReturnPart,
)
from pydantic_core import to_json
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from structlog import get_logger

from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.state import SearchState

logger = get_logger(__name__)


# What one tool result may weigh in a thread's history, as the model reads it (the JSON text of its content).
# A grouped aggregation over every subscription is hundreds of KB; replayed on every later call of the thread
# it made each of them run out of memory before doing any work.
MAX_HISTORY_TOOL_RESULT_CHARS = 20_000
# The artifact a result carries is for the client, which got it in the turn that made it. Over this size a query
# artifact keeps only its reference (the rows it rendered go); any other artifact that large is dropped.
MAX_HISTORY_TOOL_METADATA_CHARS = 10_000
# What a query artifact carries of its rows: the rendered table or chart, and the chart's points.
_QUERY_ARTIFACT_ROWS = frozenset({"rendered_block", "chart"})


def _truncation_marker(original_chars: int) -> str:
    return (
        f"\n[truncated from {original_chars} chars when stored in the conversation history; "
        "re-run the query to see all rows]"
    )


def _bounded_part(part: ModelRequestPart) -> ModelRequestPart:
    """A tool result within the limits of the history; any other part, and a result within them, as it is.

    Over the limit the content becomes the start of its JSON text and a marker that says how long it was, the
    whole no longer than the limit, so bounding what is bounded already changes nothing. The call id and tool
    name stay: the model's provider wants every call answered.
    """
    match part:
        case ToolReturnPart(metadata=metadata):
            text = part.model_response_str()
            kept_metadata = _bounded_metadata(metadata)
            if len(text) <= MAX_HISTORY_TOOL_RESULT_CHARS:
                return part if kept_metadata is metadata else replace(part, metadata=kept_metadata)
            logger.info(
                "Truncated a large tool result in the conversation history",
                tool_name=part.tool_name,
                original_chars=len(text),
            )
            marker = _truncation_marker(len(text))
            # The typed tool parts (a tool search's matches) declare their own content shape; they are small, and
            # it is only ever the text of a large result that is put in place of its content.
            plain = cast("ToolReturnPart[Any]", part)
            return replace(
                plain, content=text[: MAX_HISTORY_TOOL_RESULT_CHARS - len(marker)] + marker, metadata=kept_metadata
            )
        case _:
            return part


def _small(metadata: object) -> bool:
    return len(to_json(metadata, fallback=repr)) <= MAX_HISTORY_TOOL_METADATA_CHARS


def _bounded_metadata(metadata: object) -> object:
    """A result's artifact within the limit: as it is when small, a query artifact as its reference, else None.

    A reopened chat still shows which query a result came from (``GET /chats/{id}`` reads the artifact back); a
    stored history holds the artifact as the model it was, or as its JSON once loaded again.
    """
    if metadata is None or _small(metadata):
        return metadata
    match metadata:
        case QueryArtifact():
            lighter: object = metadata.model_copy(update=dict.fromkeys(_QUERY_ARTIFACT_ROWS))
        case dict() if "query_id" in metadata:
            lighter = metadata | dict.fromkeys(_QUERY_ARTIFACT_ROWS)
        case _:
            lighter = None
    return lighter if lighter is not None and _small(lighter) else None


def _bounded_message(message: ModelMessage) -> ModelMessage:
    match message:
        case ModelRequest(parts=parts):  # tool results come back in requests
            return replace(message, parts=[_bounded_part(part) for part in parts])
        case _:
            return message


def truncate_tool_results(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """The messages with every tool result bounded (see ``_bounded_part``); the ones given are left as they are."""
    return [_bounded_message(message) for message in messages]


def dump_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Model messages as the JSON the state stores in ``message_history``, no tool result larger than the limit."""
    return ModelMessagesTypeAdapter.dump_python(truncate_tool_results(messages), mode="json")


def load_messages(stored: Sequence[dict[str, Any]]) -> list[ModelMessage] | None:
    """The stored ``message_history`` back as model messages; None when there is none to replay.

    A thread stored before results were bounded is repaired here, on its next call. Validation leaves a
    result's content as the object it was stored as, so a huge one is not copied before it is cut.
    """
    return truncate_tool_results(ModelMessagesTypeAdapter.validate_python(list(stored))) if stored else None


@dataclass(frozen=True)
class StoredThread:
    """A conversation as the agent keeps it: its latest state, the route that wrote it, and when."""

    thread_id: str
    agent_type: str
    updated_at: datetime
    state: SearchState


def _latest_per_thread(agent_types: Sequence[str] | None = None) -> Any:
    """The latest snapshot of every thread (of the given agent types): one row per thread, as a subquery."""
    stmt = (
        select(
            AgentRunTable.thread_id.label("thread_id"),
            AgentRunTable.agent_type.label("agent_type"),
            GraphSnapshotTable.created_at.label("updated_at"),
            GraphSnapshotTable.snapshot_data.label("snapshot_data"),
        )
        .join(AgentRunTable, GraphSnapshotTable.run_id == AgentRunTable.run_id)
        .distinct(AgentRunTable.thread_id)
        .order_by(AgentRunTable.thread_id, AgentRunTable.created_at.desc(), GraphSnapshotTable.sequence_number.desc())
    )
    if agent_types is not None:
        stmt = stmt.where(AgentRunTable.agent_type.in_(list(agent_types)))
    return stmt.subquery()


def read_state(snapshot_data: Any, thread_id: str) -> SearchState | None:
    """A snapshot's state as it reads today; None when it no longer reads at all.

    A model may have changed since the snapshot was written. The search cache (``query``) is the first thing
    let go: a reopened chat is its messages and its form, the cache only what the last turn found. A state
    that does not read without it either is left out, with a warning naming the thread.
    """
    data = dict((snapshot_data or {}).get("state") or {})
    for attempt in (data, {key: value for key, value in data.items() if key != "query"}):
        try:
            return SearchState.model_validate(attempt)
        except ValidationError as error:
            reason = error
    logger.warning("A stored thread's state no longer reads; it is left out", thread_id=thread_id, error=str(reason))
    return None


def _thread(row: Any) -> StoredThread | None:
    state = read_state(row.snapshot_data, row.thread_id)
    if state is None:
        return None
    return StoredThread(thread_id=row.thread_id, agent_type=row.agent_type, updated_at=row.updated_at, state=state)


def load_thread(session: Session, thread_id: str) -> StoredThread | None:
    """The thread's latest state with the route that wrote it; None when the thread is unknown or unreadable."""
    latest = _latest_per_thread()
    row = session.execute(select(latest).where(latest.c.thread_id == thread_id)).one_or_none()
    return _thread(row) if row is not None else None


def list_threads(session: Session, *, agent_types: Sequence[str], owner: str | None, limit: int) -> list[StoredThread]:
    """The threads of the given routes that belong to ``owner`` (None: the unowned ones), newest first."""
    latest = _latest_per_thread(agent_types)
    owner_column = latest.c.snapshot_data["state"]["owner"].astext
    stmt = (
        select(latest)
        .where(owner_column.is_(None) if owner is None else owner_column == owner)
        .order_by(latest.c.updated_at.desc())
        .limit(limit)
    )
    return [thread for row in session.execute(stmt).all() if (thread := _thread(row)) is not None]


def delete_thread(session: Session, thread_id: str) -> int:
    """Forget a thread: its snapshots and its runs. The number of runs deleted."""
    run_ids = select(AgentRunTable.run_id).where(AgentRunTable.thread_id == thread_id)
    session.execute(delete(GraphSnapshotTable).where(GraphSnapshotTable.run_id.in_(run_ids)))
    result = session.execute(delete(AgentRunTable).where(AgentRunTable.thread_id == thread_id))
    session.commit()
    return int(getattr(result, "rowcount", 0) or 0)


class PostgresStatePersistence:
    """PostgreSQL state persistence for search agent.

    Stores state snapshots in the graph_snapshots table, allowing execution to be
    interrupted and resumed across conversation turns.
    Uses thread_id for persistence so state continues across multiple runs in the same thread.
    """

    def __init__(self, thread_id: str, run_id: UUID, session: Session):
        self.thread_id = thread_id
        self.run_id = run_id
        self.session = session
        self._sequence_counter = 0

    async def snapshot(self, state: SearchState) -> None:
        """Save a state snapshot to PostgreSQL.

        Args:
            state: Current agent state
        """
        snapshot_data = {
            "state": state.model_dump(mode="json"),
            "sequence": self._sequence_counter,
        }

        db_snapshot = GraphSnapshotTable(
            run_id=self.run_id,
            sequence_number=self._sequence_counter,
            snapshot_data=snapshot_data,
        )

        self.session.add(db_snapshot)
        self.session.flush()

        logger.debug(
            "Saved state snapshot",
            run_id=str(self.run_id),
            sequence=self._sequence_counter,
        )

        self._sequence_counter += 1

    async def load_state(self) -> SearchState | None:
        """Load the most recent state for this thread.

        Returns:
            The deserialized SearchState, or None if no snapshots exist
        """
        thread = await self.load_thread()
        return thread.state if thread is not None else None

    async def load_thread(self) -> StoredThread | None:
        """The thread's most recent state, with the route that wrote it; None if no snapshots exist."""
        stmt = (
            select(GraphSnapshotTable, AgentRunTable.agent_type)
            .join(AgentRunTable, GraphSnapshotTable.run_id == AgentRunTable.run_id)
            .where(AgentRunTable.thread_id == self.thread_id)
            .order_by(AgentRunTable.created_at.desc(), GraphSnapshotTable.sequence_number.desc())
            .limit(1)
        )

        row = self.session.execute(stmt).one_or_none()
        if row is None:
            logger.debug("No snapshots found for thread", thread_id=self.thread_id)
            return None
        db_snapshot, agent_type = row

        state = read_state(db_snapshot.snapshot_data, self.thread_id)
        if state is None:  # the thread starts over rather than never answering again
            return None

        # Update sequence counter to continue from where we left off
        self._sequence_counter = db_snapshot.sequence_number + 1

        logger.debug(
            "Loaded state for resume",
            thread_id=self.thread_id,
            run_id=str(self.run_id),
            sequence=db_snapshot.sequence_number,
        )

        return StoredThread(
            thread_id=self.thread_id, agent_type=agent_type, updated_at=db_snapshot.created_at, state=state
        )
