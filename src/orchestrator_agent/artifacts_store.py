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

"""Minimal persistence for artifacts emitted during an A2A turn.

A declarative kagent orchestrator text-extracts a sub-agent's response and
drops structured ``DataPart``s, so the artifact *reference* can't ride the A2A
stream to the client. Instead we persist each emitted artifact keyed by
``thread_id`` (the A2A context id); a surface (the proxy) reads them back at
end-of-turn via ``GET /threads/{thread_id}/queries`` and injects the
``wfo-artifact:`` fences. This is the side channel that bypasses the LLM.

Self-contained SQLAlchemy Core table (its own ``MetaData``) so it isn't tied
to orchestrator-core's migrations; ``ensure_table()`` creates it on startup.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import Column, DateTime, MetaData, String, Table, select
from sqlalchemy.dialects.postgresql import JSONB

logger = structlog.get_logger(__name__)

artifacts_metadata = MetaData()

emitted_artifacts = Table(
    "wfo_emitted_artifacts",
    artifacts_metadata,
    Column("id", String, primary_key=True),
    Column("thread_id", String, index=True, nullable=False),
    Column("artifact_name", String, nullable=False),
    Column("artifact_data", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def ensure_table() -> None:
    """Create the artifacts table if it doesn't exist (called on startup)."""
    from orchestrator.db import db

    artifacts_metadata.create_all(bind=db.session.get_bind(), checkfirst=True)


def record_artifact(thread_id: str, name: str, data: dict[str, Any]) -> None:
    """Persist one emitted artifact descriptor for later end-of-turn pickup."""
    from orchestrator.db import db

    db.session.execute(
        emitted_artifacts.insert().values(
            id=str(uuid4()),
            thread_id=thread_id,
            artifact_name=name,
            artifact_data=data,
            created_at=datetime.now(timezone.utc),
        )
    )
    db.session.commit()


def list_artifacts(thread_id: str) -> list[dict[str, Any]]:
    """Return artifact descriptors emitted under ``thread_id``, oldest first."""
    from orchestrator.db import db

    rows = db.session.execute(
        select(emitted_artifacts.c.artifact_name, emitted_artifacts.c.artifact_data)
        .where(emitted_artifacts.c.thread_id == thread_id)
        .order_by(emitted_artifacts.c.created_at)
    ).all()
    return [{"name": r.artifact_name, "data": r.artifact_data} for r in rows]
