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

"""A stored thread whose state no longer reads does not take the others down with it."""

import os
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.persistence import list_threads, read_state  # noqa: E402

STALE_QUERY = {"aggregate": {"aggregations": []}}  # a shape a later model no longer takes


def test_a_state_that_reads_is_read_whole():
    state = read_state({"state": {"owner": "alice"}}, "t1")
    assert state is not None and state.owner == "alice"


def test_a_stale_search_cache_is_let_go():
    state = read_state({"state": {"owner": "alice", "query": STALE_QUERY}}, "t1")
    assert state is not None and state.owner == "alice" and state.query is None


def test_a_state_that_does_not_read_at_all_is_left_out():
    assert read_state({"state": {"message_history": "not a list"}}, "t1") is None
    assert read_state(None, "t1") is not None  # no snapshot data: a fresh state


def test_the_list_skips_a_thread_it_cannot_read():
    rows = [
        SimpleNamespace(
            thread_id="good",
            agent_type="agui",
            updated_at=datetime(2026, 10, 20, tzinfo=UTC),
            snapshot_data={"state": {}},
        ),
        SimpleNamespace(
            thread_id="bad",
            agent_type="agui",
            updated_at=datetime(2026, 10, 19, tzinfo=UTC),
            snapshot_data={"state": {"message_history": 3}},
        ),
    ]
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    assert [thread.thread_id for thread in list_threads(session, agent_types=("agui",), owner=None, limit=10)] == [
        "good"
    ]
