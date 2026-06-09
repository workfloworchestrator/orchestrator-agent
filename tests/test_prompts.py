"""Tests for capability instruction / framing generation."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.capabilities.prompts import (
    BASE_FRAMING,
    get_aggregation_instructions,
    get_entity_instructions,
    get_export_instructions,
    get_search_instructions,
)


class TestBaseFraming:
    def test_text_only_rendering_rules(self):
        assert "Markdown" in BASE_FRAMING
        # Charts/tables are injected by the adapter; the framing tells the model not to draw them.
        assert "chart or table" in BASE_FRAMING
        assert "automatically" in BASE_FRAMING


class TestSearchInstructions:
    @pytest.mark.parametrize(
        "required",
        [
            pytest.param(["Searching", "search", "discover_filter_paths", "get_valid_operators"], id="key-elements"),
            pytest.param(["Filtering Rules", "MANDATORY FIRST STEP"], id="filtering-rules"),
            pytest.param(["PREFER LENIENT OPERATORS", "like", "between"], id="lenient-operators"),
            pytest.param(["automatically broadens", "fallback_used"], id="auto-broaden-note"),
            pytest.param(["KEEP KNOWN STRUCTURED FILTERS", "status", "product"], id="structured-filter-retention"),
            pytest.param(["EXTRACT IDENTIFIERS", "highest-signal"], id="identifier-extraction"),
        ],
    )
    def test_instructions_contain(self, required):
        text = get_search_instructions()
        missing = [snippet for snippet in required if snippet not in text]
        assert not missing, f"missing from search instructions: {missing}"


class TestAggregationInstructions:
    def test_contains_key_elements(self):
        text = get_aggregation_instructions()
        assert "Aggregating" in text
        assert "aggregate" in text
        assert "group_by" in text
        assert "temporal_group_by" in text

    def test_includes_filtering_rules(self):
        assert "Filtering Rules" in get_aggregation_instructions()

    def test_no_semantic_fallback_note(self):
        text = get_aggregation_instructions()
        assert "automatically retries with a broader semantic search" not in text
        assert "PREFER LENIENT OPERATORS" in text


class TestEntityInstructions:
    def test_contains_key_elements(self):
        text = get_entity_instructions()
        assert "resolve_entity" in text
        assert "get_entity_details" in text
        assert "id-prefix" in text
        assert "NOT an export" in text


class TestExportInstructions:
    def test_contains_key_elements(self):
        text = get_export_instructions()
        assert "export_query" in text
        assert "EXPLICITLY" in text


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
        monkeypatch.setattr("orchestrator_agent.capabilities.prompts.agent_settings.AGENT_DOMAIN_CONTEXT", context)
        text = get_search_instructions()
        assert ("## Domain Knowledge" in text) is expect_section
        if expect_section:
            assert context.strip() in text


class TestAutoBroadenNote:
    def test_search_instructions_mention_fallback_used(self):
        text = get_search_instructions()
        assert "automatically broadens" in text
        assert "fallback_used" in text
        # retriever selection is the tool's job now — the prompt must not instruct it.
        assert "CHOOSE A RETRIEVER" not in text
