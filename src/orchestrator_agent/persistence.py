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
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from orchestrator.core.db.models import AgentRunTable, GraphSnapshotTable
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from structlog import get_logger

from orchestrator_agent.state import SearchState

logger = get_logger(__name__)


def dump_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Model messages as the JSON the state stores in ``message_history``."""
    return ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")


def load_messages(stored: Sequence[dict[str, Any]]) -> list[ModelMessage] | None:
    """The stored ``message_history`` back as model messages; None when there is none to replay."""
    return ModelMessagesTypeAdapter.validate_python(list(stored)) if stored else None


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
