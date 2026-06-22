"""Tests for assembled plugin instructions and the agent system prompt."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.capabilities.loader import load_plugin_specs, load_system_prompt

_DOMAIN = "orchestrator_agent.capabilities.loader.agent_settings.AGENT_DOMAIN_CONTEXT"


def _instructions(plugin_id: str) -> str:
    return {s.id: s.instructions for s in load_plugin_specs()}[plugin_id]


class TestSystemPrompt:
    def test_text_only_rendering_rules(self):
        system_prompt = load_system_prompt()
        assert "Markdown" in system_prompt
        # The per-call "don't restate the rows" rule is injected with the result (base.py
        # _injection_notice). The system prompt states the global text-only format and points the
        # model at that per-call note, rather than duplicating the detail.
        assert "chart or table" in system_prompt
        assert "automatically" not in system_prompt


class TestSearchInstructions:
    def test_contains_intent_elements(self):
        # Thin, intent-only prompt: how to filter/choose operators lives in the MCP tool descriptions.
        text = _instructions("search")
        for snippet in ("Searching", "Run the search", "entity_type"):
            assert snippet in text

    def test_filtering_mechanics_moved_to_server(self):
        # The filter-building rules now live in orchestrator-core's tool descriptions, not the prompt.
        text = _instructions("search")
        for moved in ("PREFER LENIENT OPERATORS", "Filtering Rules", "MANDATORY FIRST STEP", "fallback_used"):
            assert moved not in text

    def test_prompts_do_not_name_tools(self):
        # Prompts describe intent; the model binds to a tool from its description. No tool names,
        # no leftover ${...} placeholders.
        text = _instructions("search")
        assert "${" not in text
        for tool_name in ("discover_filter_paths", "get_valid_operators"):
            assert tool_name not in text


class TestAggregationInstructions:
    def test_contains_intent_elements(self):
        text = _instructions("aggregate")
        assert "Aggregating" in text
        assert "Run the aggregation" in text
        assert "group_by" in text
        assert "temporal_group_by" in text

    def test_filtering_mechanics_moved_to_server(self):
        text = _instructions("aggregate")
        for moved in ("Filtering Rules", "PREFER LENIENT OPERATORS", "MANDATORY FIRST STEP"):
            assert moved not in text


class TestEntityInstructions:
    def test_contains_key_elements(self):
        text = _instructions("entity")
        assert "resolve it" in text
        assert "id-prefix" in text
        assert "NOT an export" in text


class TestExportInstructions:
    def test_contains_key_elements(self):
        text = _instructions("export")
        assert "downloadable export" in text
        assert "EXPLICITLY" in text


class TestDomainContextSection:
    """Operator domain knowledge is injected once into the agent-level system prompt, not per-plugin."""

    @pytest.mark.parametrize(
        "context, expect_section",
        [
            pytest.param("IS#### maps to imsCircuitId", True, id="set"),
            pytest.param("", False, id="empty"),
            pytest.param("   ", False, id="whitespace-only"),
        ],
    )
    def test_domain_section_in_system_prompt(self, monkeypatch, context, expect_section):
        monkeypatch.setattr(_DOMAIN, context)
        system_prompt = load_system_prompt()
        assert ("## Domain Knowledge" in system_prompt) is expect_section
        if expect_section:
            assert context.strip() in system_prompt

    def test_domain_context_not_in_plugin_instructions(self, monkeypatch):
        # The block lives in the system prompt now — no plugin body should carry it.
        monkeypatch.setattr(_DOMAIN, "IS#### maps")
        assert "## Domain Knowledge" not in _instructions("search")
        assert "## Domain Knowledge" not in _instructions("aggregate")
