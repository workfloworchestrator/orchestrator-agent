"""Tests for prompt generation functions."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

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
    @pytest.mark.parametrize(
        "required",
        [
            pytest.param(["Searching", "run_search", "discover_filter_paths", "set_filter_tree"], id="key-elements"),
            pytest.param(["Filtering Rules", "MANDATORY FIRST STEP"], id="filtering-rules"),
            pytest.param(["PREFER LENIENT OPERATORS", "like", "between"], id="lenient-operators"),
            pytest.param(["automatically retries with a broader semantic search"], id="semantic-fallback-note"),
            pytest.param(["KEEP KNOWN STRUCTURED FILTERS", "status", "product"], id="structured-filter-retention"),
            pytest.param(["EXTRACT IDENTIFIERS", "highest-signal"], id="identifier-extraction"),
        ],
    )
    def test_prompt_contains(self, required):
        prompt = get_search_execution_prompt(_make_state())
        missing = [snippet for snippet in required if snippet not in prompt]
        assert not missing, f"missing from search prompt: {missing}"


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

    def test_aggregation_has_no_semantic_fallback_note(self):
        prompt = get_aggregation_execution_prompt(_make_state())
        assert "automatically retries with a broader semantic search" not in prompt
        assert "PREFER LENIENT OPERATORS" in prompt


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

    def test_mentions_id_prefix_single_result_actions_task(self):
        prompt = get_planning_prompt(_make_state())
        assert "id-prefix" in prompt
        assert "RESULT_ACTIONS" in prompt


class TestGetResultActionsPrompt:
    def test_contains_key_elements(self):
        prompt = get_result_actions_prompt(_make_state())
        assert "Acting on Results" in prompt
        assert "prepare_export" in prompt
        assert "fetch_entity_details" in prompt

    def test_mentions_get_entity_by_id_for_id_or_prefix(self):
        prompt = get_result_actions_prompt(_make_state())
        assert "get_entity_by_id" in prompt
        assert "id-prefix" in prompt
