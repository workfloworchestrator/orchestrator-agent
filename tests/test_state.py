"""Tests for state models — ExecutionPlan, Task, TaskAction, TaskStatus, SearchState."""

from __future__ import annotations

import os
from uuid import uuid4

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.state import ExecutionPlan, SearchState, Task, TaskAction, TaskStatus


class TestTaskAction:
    def test_values(self):
        assert TaskAction.SEARCH.value == "search"
        assert TaskAction.AGGREGATION.value == "aggregation"
        assert TaskAction.RESULT_ACTIONS.value == "result_actions"
        assert TaskAction.TEXT_RESPONSE.value == "text_response"


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.EXECUTING.value == "executing"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestTask:
    def test_defaults(self):
        task = Task(action_type=TaskAction.SEARCH, reasoning="Find subs")
        assert task.status == TaskStatus.PENDING

    def test_status_excluded_from_serialization(self):
        task = Task(action_type=TaskAction.SEARCH, reasoning="Find subs")
        dumped = task.model_dump()
        assert "status" not in dumped


class TestExecutionPlan:
    def test_creation(self):
        plan = ExecutionPlan(
            tasks=[
                Task(action_type=TaskAction.SEARCH, reasoning="Find subs"),
                Task(action_type=TaskAction.RESULT_ACTIONS, reasoning="Export"),
            ]
        )
        assert len(plan.tasks) == 2

    def test_empty_plan(self):
        plan = ExecutionPlan(tasks=[])
        assert plan.tasks == []


class TestSearchState:
    def test_defaults(self):
        state = SearchState()
        assert state.user_input == ""
        assert state.run_id is None
        assert state.query_id is None
        assert state.query is None
        assert state.pending_filters is None

    def test_with_values(self):
        rid = uuid4()
        qid = uuid4()
        state = SearchState(user_input="test", run_id=rid, query_id=qid)
        assert state.user_input == "test"
        assert state.run_id == rid
        assert state.query_id == qid

    def test_memory_initialized(self):
        state = SearchState(user_input="test")
        assert state.memory is not None
        assert state.memory.turns == []

    def test_serialization_roundtrip(self):
        state = SearchState(user_input="test query")
        data = state.model_dump(mode="json")
        restored = SearchState.model_validate(data)
        assert restored.user_input == "test query"
