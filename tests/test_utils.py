"""Tests for utility functions."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from unittest.mock import MagicMock

from orchestrator_agent.state import ExecutionPlan, Task, TaskAction, TaskStatus
from orchestrator_agent.utils import current_timestamp_ms, log_agent_request, log_execution_plan


class TestCurrentTimestampMs:
    def test_returns_int(self):
        ts = current_timestamp_ms()
        assert isinstance(ts, int)

    def test_is_reasonable(self):
        ts = current_timestamp_ms()
        # Should be roughly current time in ms (after 2020-01-01)
        assert ts > 1_577_836_800_000


class TestLogExecutionPlan:
    def test_none_plan(self, capsys):
        log_execution_plan(None)
        captured = capsys.readouterr()
        assert "[EXECUTION PLAN] None" in captured.out

    def test_plan_with_tasks(self, capsys):
        plan = ExecutionPlan(
            tasks=[
                Task(action_type=TaskAction.SEARCH, reasoning="Find subs", status=TaskStatus.PENDING),
                Task(action_type=TaskAction.RESULT_ACTIONS, reasoning="Export", status=TaskStatus.COMPLETED),
            ]
        )
        log_execution_plan(plan)
        captured = capsys.readouterr()
        assert "[EXECUTION PLAN] 2 tasks" in captured.out
        assert "PENDING" in captured.out
        assert "COMPLETED" in captured.out
        assert "search" in captured.out
        assert "Find subs" in captured.out


class TestLogAgentRequest:
    def test_with_messages(self, capsys):
        msg = MagicMock()
        msg.kind = "request"
        part = MagicMock()
        part.__class__.__name__ = "UserPromptPart"
        part.content = "show subscriptions"
        msg.parts = [part]

        log_agent_request("Search", "You are a search agent", [msg])
        captured = capsys.readouterr()
        assert "[Search] LLM Request" in captured.out
        assert "[INSTRUCTIONS]" in captured.out
        assert "You are a search agent" in captured.out
        assert "[MESSAGE HISTORY] (1 messages)" in captured.out
        assert "show subscriptions" in captured.out

    def test_empty_messages(self, capsys):
        log_agent_request("Planner", "Plan things", [])
        captured = capsys.readouterr()
        assert "[MESSAGE HISTORY] (empty)" in captured.out

    def test_part_without_content(self, capsys):
        msg = MagicMock()
        msg.kind = "request"
        part = MagicMock(spec=[])  # no content attribute
        part.__class__.__name__ = "ToolCallPart"
        del part.content  # ensure hasattr returns False
        msg.parts = [part]

        log_agent_request("Search", "instructions", [msg])
        captured = capsys.readouterr()
        assert "[ToolCallPart]" in captured.out
