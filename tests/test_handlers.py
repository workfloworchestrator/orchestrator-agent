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
