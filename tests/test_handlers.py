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

"""Tests for handler functions — validation logic (no DB needed)."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.handlers import execute_aggregation_with_persistence, execute_search_with_persistence


class TestExecuteSearchWithPersistence:
    async def test_raises_when_run_id_is_none(self):
        with pytest.raises(ValueError, match="run_id is required"):
            await execute_search_with_persistence(query=None, db_session=None, run_id=None)


class TestExecuteAggregationWithPersistence:
    async def test_raises_when_run_id_is_none(self):
        with pytest.raises(ValueError, match="run_id is required"):
            await execute_aggregation_with_persistence(query=None, db_session=None, run_id=None)
