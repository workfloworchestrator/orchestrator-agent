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
from orchestrator_agent.settings import SearchEffort
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
            pytest.param(["automatically retries"], id="semantic-fallback-note"),
            pytest.param(["KEEP KNOWN STRUCTURED FILTERS", "status", "product"], id="structured-filter-retention"),
            pytest.param(["EXTRACT IDENTIFIERS", "highest-signal"], id="identifier-extraction"),
            pytest.param(["customer/organisation", "PRODUCT NAME vs TYPE"], id="customer-and-product-guidance"),
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
        assert "automatically retries" not in prompt
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


class TestDomainContextSection:
    @pytest.mark.parametrize(
        "context, expect_section",
        [
            pytest.param("IS#### maps to imsCircuitId", True, id="set"),
            pytest.param("", False, id="empty"),
            pytest.param("   ", False, id="whitespace-only"),
        ],
    )
    def test_domain_section(self, monkeypatch, context, expect_section):
        monkeypatch.setattr("orchestrator_agent.prompts.agent_settings.AGENT_DOMAIN_CONTEXT", context)
        prompt = get_search_execution_prompt(_make_state())
        assert ("## Domain Knowledge" in prompt) is expect_section
        if expect_section:
            assert context.strip() in prompt


class TestPlannerEffortGuidance:
    @pytest.mark.parametrize(
        "effort, expect_clarify",
        [
            pytest.param(SearchEffort.HIGH, False, id="high-proceeds"),
            pytest.param(SearchEffort.MEDIUM, True, id="medium-asks-when-ambiguous"),
            pytest.param(SearchEffort.LOW, True, id="low-prefers-asking"),
        ],
    )
    def test_planner_clarify_bias(self, monkeypatch, effort, expect_clarify):
        monkeypatch.setattr("orchestrator_agent.prompts.agent_settings.AGENT_SEARCH_EFFORT", effort)
        prompt = get_planning_prompt(_make_state())
        assert ("clarifying question" in prompt) is expect_clarify


class TestEmptyResultsGuidance:
    @pytest.mark.parametrize(
        "effort, expect_auto_retry",
        [
            pytest.param(SearchEffort.HIGH, True, id="high-auto-retries"),
            pytest.param(SearchEffort.MEDIUM, True, id="medium-auto-retries"),
            pytest.param(SearchEffort.LOW, False, id="low-asks-instead"),
        ],
    )
    def test_empty_results_note(self, monkeypatch, effort, expect_auto_retry):
        monkeypatch.setattr("orchestrator_agent.prompts.agent_settings.AGENT_SEARCH_EFFORT", effort)
        prompt = get_search_execution_prompt(_make_state())
        assert ("automatically retries" in prompt) is expect_auto_retry
        if not expect_auto_retry:
            assert "ask whether to broaden the search" in prompt


class TestRetrieverGuidance:
    @pytest.mark.parametrize(
        "embeddings_enabled, expect_hybrid",
        [
            pytest.param(True, True, id="embeddings-on"),
            pytest.param(False, False, id="embeddings-off"),
        ],
    )
    def test_retriever_guidance(self, monkeypatch, embeddings_enabled, expect_hybrid):
        monkeypatch.setattr("orchestrator_agent.prompts.llm_settings.EMBEDDING_API_ENABLED", embeddings_enabled)
        prompt = get_search_execution_prompt(_make_state())
        assert "CHOOSE A RETRIEVER" in prompt
        if expect_hybrid:
            assert "retriever=HYBRID" in prompt
            assert "cannot match opaque tokens" in prompt
        else:
            assert "retriever=FUZZY" in prompt
            assert "retriever=HYBRID" not in prompt
