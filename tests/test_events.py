"""Tests for event factories and RunContext."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.events import (
    AgentStepActiveEvent,
    PlanCreatedEvent,
    RunContext,
    make_plan_created_event,
    make_step_active_event,
)
from orchestrator_agent.state import SearchState


class TestMakeStepActiveEvent:
    def test_basic(self):
        event = make_step_active_event("Search")
        assert isinstance(event, AgentStepActiveEvent)
        assert event.name == "AGENT_STEP_ACTIVE"
        assert event.value["step"] == "Search"
        assert "reasoning" not in event.value
        assert event.timestamp > 0

    def test_with_reasoning(self):
        event = make_step_active_event("Search", reasoning="Looking for subscriptions")
        assert event.value["step"] == "Search"
        assert event.value["reasoning"] == "Looking for subscriptions"


class TestMakePlanCreatedEvent:
    def test_basic(self):
        tasks = [
            {"skill_name": "Search", "reasoning": "Find subscriptions"},
            {"skill_name": "Result Actions", "reasoning": "Export results"},
        ]
        event = make_plan_created_event(tasks)
        assert isinstance(event, PlanCreatedEvent)
        assert event.name == "PLAN_CREATED"
        assert len(event.value) == 2
        assert event.value[0]["skill_name"] == "Search"
        assert event.timestamp > 0

    def test_empty_tasks(self):
        event = make_plan_created_event([])
        assert event.value == []


class TestRunContext:
    def test_holds_state(self):
        state = SearchState(user_input="test")
        ctx = RunContext(state=state)
        assert ctx.state.user_input == "test"
