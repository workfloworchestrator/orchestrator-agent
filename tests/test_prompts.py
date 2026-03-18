"""Tests for prompt generation functions."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.prompts import (
    get_aggregation_execution_prompt,
    get_planning_prompt,
    get_result_actions_prompt,
    get_search_execution_prompt,
    get_text_response_prompt,
)
from orchestrator_agent.state import SearchState


def _make_state(user_input: str = "show subscriptions") -> SearchState:
    return SearchState(user_input=user_input)


class TestGetSearchExecutionPrompt:
    def test_contains_key_elements(self):
        prompt = get_search_execution_prompt(_make_state())
        assert "Searching" in prompt
        assert "run_search" in prompt
        assert "discover_filter_paths" in prompt
        assert "set_filter_tree" in prompt

    def test_includes_filtering_rules(self):
        prompt = get_search_execution_prompt(_make_state())
        assert "Filtering Rules" in prompt
        assert "MANDATORY FIRST STEP" in prompt


class TestGetAggregationExecutionPrompt:
    def test_contains_key_elements(self):
        prompt = get_aggregation_execution_prompt(_make_state())
        assert "Aggregating" in prompt
        assert "run_aggregation" in prompt
        assert "set_temporal_grouping" in prompt
        assert "set_grouping" in prompt
        assert "set_aggregations" in prompt

    def test_includes_filtering_rules(self):
        prompt = get_aggregation_execution_prompt(_make_state())
        assert "Filtering Rules" in prompt


class TestGetTextResponsePrompt:
    def test_contains_key_elements(self):
        prompt = get_text_response_prompt(_make_state())
        assert "Responding" in prompt
        assert "Available Capabilities" in prompt
        assert "SUBSCRIPTION" in prompt


class TestGetPlanningPrompt:
    def test_contains_key_elements(self):
        prompt = get_planning_prompt(_make_state())
        assert "Execution Planning" in prompt
        assert "Break into tasks" in prompt
        assert "RESULT_ACTIONS" in prompt

    def test_no_redundant_tasks_warning(self):
        prompt = get_planning_prompt(_make_state())
        assert "Do NOT create redundant tasks" in prompt


class TestGetResultActionsPrompt:
    def test_contains_key_elements(self):
        prompt = get_result_actions_prompt(_make_state())
        assert "Acting on Results" in prompt
        assert "prepare_export" in prompt
        assert "fetch_entity_details" in prompt
