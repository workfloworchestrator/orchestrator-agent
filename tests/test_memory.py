"""Tests for Memory, Turn, Step dataclasses, and collect_tool_descriptions."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse

from orchestrator_agent.memory import (
    FALLBACK_MESSAGE,
    AgentStep,
    Memory,
    MemoryScope,
    Step,
    ToolStep,
    Turn,
    collect_tool_descriptions,
)
from orchestrator_agent.state import SearchState


class TestStep:
    def test_defaults(self):
        s = Step(step_type="search", description="Searched")
        assert s.success is True
        assert s.error_message is None
        assert s.timestamp is not None

    def test_failed_step(self):
        s = Step(step_type="search", description="Failed", success=False, error_message="boom")
        assert s.success is False
        assert s.error_message == "boom"


class TestAgentStep:
    def test_add_tool_step(self):
        agent_step = AgentStep(step_type="Search", description="Executing Search")
        tool = ToolStep(step_type="run_search", description="Searched 5 subscriptions")
        agent_step.add_tool_step(tool)
        assert len(agent_step.tool_steps) == 1
        assert agent_step.tool_steps[0].description == "Searched 5 subscriptions"

    def test_tool_step_with_context(self):
        ts = ToolStep(step_type="run_search", description="Searched", context={"query_id": "q1"})
        assert ts.context["query_id"] == "q1"


class TestTurn:
    def test_is_complete_false(self):
        turn = Turn(user_question="hello")
        assert turn.is_complete is False

    def test_is_complete_true(self):
        turn = Turn(user_question="hello", assistant_answer="hi")
        assert turn.is_complete is True


class TestCollectToolDescriptions:
    def test_with_descriptions(self):
        steps = [
            AgentStep(
                step_type="Search",
                description="Executing Search",
                tool_steps=[
                    ToolStep(step_type="run_search", description="Searched 5 subscriptions"),
                    ToolStep(step_type="set_filter", description="Applied status filter"),
                ],
            )
        ]
        result = collect_tool_descriptions(steps)
        assert result == "Searched 5 subscriptions. Applied status filter."

    def test_empty_steps(self):
        assert collect_tool_descriptions([]) == FALLBACK_MESSAGE

    def test_no_descriptions(self):
        steps = [AgentStep(step_type="Search", description="Executing Search", tool_steps=[])]
        assert collect_tool_descriptions(steps) == FALLBACK_MESSAGE

    def test_empty_description_filtered(self):
        steps = [
            AgentStep(
                step_type="Search",
                description="Executing Search",
                tool_steps=[
                    ToolStep(step_type="run_search", description=""),
                    ToolStep(step_type="set_filter", description="Applied filter"),
                ],
            )
        ]
        result = collect_tool_descriptions(steps)
        assert result == "Applied filter."


class TestMemory:
    def test_current_turn_none_when_empty(self):
        m = Memory()
        assert m.current_turn is None

    def test_current_turn_returns_incomplete(self):
        m = Memory()
        m.start_turn("hello")
        assert m.current_turn is not None
        assert m.current_turn.user_question == "hello"

    def test_current_turn_none_after_complete(self):
        m = Memory()
        m.start_turn("hello")
        m.start_step("Search")
        m.finish_step()
        m.complete_turn("done")
        assert m.current_turn is None

    def test_completed_turns(self):
        m = Memory()
        m.start_turn("q1")
        m.start_step("s1")
        m.finish_step()
        m.complete_turn("a1")
        m.start_turn("q2")
        assert len(m.completed_turns) == 1
        assert m.completed_turns[0].user_question == "q1"

    def test_start_step_raises_without_turn(self):
        m = Memory()
        with pytest.raises(ValueError, match="No active turn"):
            m.start_step("Search")

    def test_start_step_finishes_previous(self):
        m = Memory()
        m.start_turn("hello")
        m.start_step("Search")
        m.start_step("Aggregate")
        # Previous step should have been finished and added
        assert len(m.current_turn.steps) == 1
        assert m.current_turn.steps[0].step_type == "Search"
        assert m.current_turn.current_step.step_type == "Aggregate"

    def test_record_tool_step_raises_without_turn(self):
        m = Memory()
        with pytest.raises(ValueError, match="No active turn"):
            m.record_tool_step(ToolStep(step_type="run_search", description="test"))

    def test_record_tool_step_raises_without_step(self):
        m = Memory()
        m.start_turn("hello")
        with pytest.raises(ValueError, match="No active step"):
            m.record_tool_step(ToolStep(step_type="run_search", description="test"))

    def test_finish_step_raises_without_turn(self):
        m = Memory()
        with pytest.raises(ValueError, match="No active turn"):
            m.finish_step()

    def test_finish_step_raises_without_active_step(self):
        m = Memory()
        m.start_turn("hello")
        with pytest.raises(ValueError, match="No active step"):
            m.finish_step()

    def test_complete_turn_raises_without_turn(self):
        m = Memory()
        with pytest.raises(ValueError, match="No active turn"):
            m.complete_turn("done")

    def test_complete_turn_finishes_in_progress_step(self):
        m = Memory()
        m.start_turn("hello")
        m.start_step("Search")
        m.record_tool_step(ToolStep(step_type="run_search", description="Searched"))
        m.complete_turn("done")
        # Step should have been auto-finished
        assert len(m.turns[0].steps) == 1
        assert m.turns[0].is_complete

    def test_get_message_history_empty(self):
        m = Memory()
        messages = m.get_message_history()
        assert messages == []

    def test_get_message_history_with_completed_turns(self):
        m = Memory()
        m.start_turn("q1")
        m.start_step("Search")
        m.finish_step()
        m.complete_turn("a1")
        messages = m.get_message_history()
        # UserPrompt(q1), SystemPrompt(execution trace for Search step), ModelResponse(a1)
        assert len(messages) == 3
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[-1], ModelResponse)

    def test_get_message_history_no_steps_gives_two_messages(self):
        m = Memory()
        m.start_turn("q1")
        m.complete_turn("a1")
        messages = m.get_message_history()
        # UserPrompt(q1), ModelResponse(a1) — no context since no steps
        assert len(messages) == 2
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[1], ModelResponse)

    def test_get_message_history_includes_current_turn(self):
        m = Memory()
        m.start_turn("q1")
        m.complete_turn("a1")
        m.start_turn("q2")
        messages = m.get_message_history()
        # Should have: UserPrompt(q1), ModelResponse(a1), UserPrompt(q2)
        assert len(messages) == 3

    def test_get_message_history_max_turns(self):
        m = Memory()
        for i in range(10):
            m.start_turn(f"q{i}")
            m.complete_turn(f"a{i}")
        messages = m.get_message_history(max_turns=2)
        # 2 completed turns * 2 messages each = 4
        assert len(messages) == 4

    def test_get_message_history_full_scope_with_steps(self):
        m = Memory()
        m.start_turn("q1")
        m.start_step("Search")
        m.record_tool_step(ToolStep(step_type="run_search", description="Searched 5", context={"query_id": "q-1"}))
        m.finish_step()
        m.complete_turn("a1")
        messages = m.get_message_history(scope=MemoryScope.FULL)
        # UserPrompt, SystemPrompt (execution trace), ModelResponse
        assert len(messages) == 3

    def test_get_message_history_lightweight_scope(self):
        m = Memory()
        m.start_turn("q1")
        m.start_step("Search")
        m.record_tool_step(
            ToolStep(
                step_type="run_search",
                description="Searched 5",
                context={"query_id": "q-1", "query_snapshot": {"entity_type": "subscription"}},
            )
        )
        m.finish_step()
        m.complete_turn("a1")
        messages = m.get_message_history(scope=MemoryScope.LIGHTWEIGHT)
        # UserPrompt, SystemPrompt (query summary with full JSON), ModelResponse
        assert len(messages) == 3

    def test_get_message_history_minimal_scope(self):
        m = Memory()
        m.start_turn("q1")
        m.start_step("Search")
        m.record_tool_step(ToolStep(step_type="run_search", description="Searched 5", context={"query_id": "q-1"}))
        m.finish_step()
        m.complete_turn("a1")
        messages = m.get_message_history(scope=MemoryScope.MINIMAL)
        # UserPrompt, SystemPrompt (query_id + one-liner), ModelResponse
        assert len(messages) == 3

    def test_format_query_summary_no_queries(self):
        m = Memory()
        result = m._format_query_summary([])
        assert result is None

    def test_format_query_summary_with_full_query(self):
        steps = [
            AgentStep(
                step_type="Search",
                description="Executing Search",
                tool_steps=[
                    ToolStep(
                        step_type="run_search",
                        description="Searched 5",
                        context={"query_id": "q-1", "query_snapshot": {"entity_type": "subscription"}},
                    )
                ],
            )
        ]
        m = Memory()
        result = m._format_query_summary(steps, include_full_query=True)
        assert "Query q-1" in result
        assert "subscription" in result

    def test_format_query_summary_minimal(self):
        steps = [
            AgentStep(
                step_type="Search",
                description="Executing Search",
                tool_steps=[ToolStep(step_type="run_search", description="Searched 5", context={"query_id": "q-1"})],
            )
        ]
        m = Memory()
        result = m._format_query_summary(steps, include_full_query=False)
        assert result == "Query q-1: Searched 5"

    def test_format_execution_trace(self):
        steps = [
            AgentStep(
                step_type="Search",
                description="Executing Search",
                tool_steps=[ToolStep(step_type="run_search", description="Searched 5", context={"query_id": "q-1"})],
            )
        ]
        m = Memory()
        result = m._format_execution_trace(steps)
        assert "Plan executed:" in result
        assert "Search" in result
        assert "run_search" in result
        assert "query: q-1" in result

    def test_format_execution_trace_filters_planner(self):
        steps = [
            AgentStep(step_type="Planner", description="Planning"),
            AgentStep(step_type="Search", description="Executing Search"),
        ]
        m = Memory()
        result = m._format_execution_trace(steps)
        assert "Planner" not in result
        assert "Search" in result

    def test_format_execution_trace_only_planner(self):
        steps = [AgentStep(step_type="Planner", description="Planning")]
        m = Memory()
        result = m._format_execution_trace(steps)
        assert result is None

    def test_format_turn_context_empty(self):
        m = Memory()
        assert m._format_turn_context([], MemoryScope.FULL) is None

    def test_format_current_steps_no_turn(self):
        m = Memory()
        assert m.format_current_steps() == "None"

    def test_format_current_steps_with_steps(self):
        m = Memory()
        m.start_turn("hello")
        m.start_step("Search")
        m.record_tool_step(ToolStep(step_type="run_search", description="Searched"))
        m.finish_step()
        result = m.format_current_steps()
        assert "Search" in result

    def test_format_current_steps_includes_in_progress(self):
        m = Memory()
        m.start_turn("hello")
        m.start_step("Search")
        result = m.format_current_steps()
        assert "Search" in result

    def test_format_context_for_llm_default(self):
        state = SearchState(user_input="test")
        result = state.memory.format_context_for_llm(state)
        assert result == ""

    def test_format_context_for_llm_with_current_steps(self):
        state = SearchState(user_input="test")
        state.memory.start_turn("test")
        state.memory.start_step("Search")
        state.memory.record_tool_step(ToolStep(step_type="run_search", description="Searched"))
        state.memory.finish_step()
        result = state.memory.format_context_for_llm(state, include_current_run_steps=True)
        assert "Steps Already Executed" in result
        assert "Search" in result
