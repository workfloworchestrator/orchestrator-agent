"""Tests for the capabilities layer: history trimming and MCP-result artifact mapping."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturn, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from orchestrator_agent.artifacts import (
    DataArtifact,
    ExportArtifact,
    QueryArtifact,
    RenderedBlock,
)
from orchestrator_agent.capabilities.hooks import (
    FilterPathGuard,
    _artifact_for,
    trim_history,
)
from orchestrator_agent.rendering.charts import aggregate_to_mermaid
from orchestrator_agent.rendering.tables import search_to_markdown
from orchestrator_agent.tool_names import (
    AGGREGATE_TOOL,
    DISCOVER_FILTER_PATHS_TOOL,
    EXPORT_QUERY_TOOL,
    RESOLVE_ENTITY_TOOL,
    SEARCH_TOOL,
)


class TestTrimHistory:
    def test_long_history_trimmed_to_window(self):
        msgs = [ModelRequest(parts=[UserPromptPart(content=str(i))]) for i in range(100)]
        trimmed = trim_history(msgs)
        assert len(trimmed) < len(msgs)
        # Most recent message is preserved.
        assert trimmed[-1] is msgs[-1]

    def test_leading_instructions_preserved(self):
        head = ModelRequest(parts=[UserPromptPart(content="start")], instructions="SYSTEM FRAMING")
        msgs = [head] + [ModelRequest(parts=[UserPromptPart(content=str(i))]) for i in range(100)]
        trimmed = trim_history(msgs)
        assert trimmed[0] is head


class TestArtifactMapping:
    def test_search_maps_to_query_artifact(self):
        # search returns `returned` + `search_type`, no `visualization` (matches live core schema).
        result = {"query_id": "q-1", "returned": 42, "results": [], "search_type": "fuzzy"}
        artifact = _artifact_for(SEARCH_TOOL, result)
        assert isinstance(artifact, QueryArtifact)
        assert artifact.query_id == "q-1"
        assert artifact.total_results == 42
        assert artifact.search_type == "fuzzy"

    def test_search_total_falls_back_to_result_count(self):
        result = {"query_id": "q-2", "results": [1, 2, 3]}
        artifact = _artifact_for(SEARCH_TOOL, result)
        assert isinstance(artifact, QueryArtifact)
        assert artifact.total_results == 3

    def test_aggregate_maps_to_query_artifact(self):
        # aggregate returns `total_results` + a string `visualization` hint (live core schema).
        result = {"query_id": "q-3", "total_results": 1, "visualization": "table", "results": [{"k": 1}]}
        artifact = _artifact_for(AGGREGATE_TOOL, result)
        assert isinstance(artifact, QueryArtifact)
        assert artifact.query_id == "q-3"
        assert artifact.total_results == 1
        assert artifact.visualization_type.type == "table"

    def test_export_maps_to_export_artifact(self):
        # export_query returns `download_path` (live core schema).
        result = {"query_id": "q-4", "download_path": "/api/search/queries/q-4/export"}
        artifact = _artifact_for(EXPORT_QUERY_TOOL, result)
        assert isinstance(artifact, ExportArtifact)
        assert artifact.download_url == "/api/search/queries/q-4/export"

    def test_entity_maps_to_data_artifact(self):
        result = {"entity_id": "e-1", "entity_type": "SUBSCRIPTION", "name": "x"}
        artifact = _artifact_for(RESOLVE_ENTITY_TOOL, result)
        assert isinstance(artifact, DataArtifact)
        assert artifact.entity_id == "e-1"
        assert artifact.entity_type == "SUBSCRIPTION"

    @pytest.mark.parametrize(
        "tool, result",
        [
            (DISCOVER_FILTER_PATHS_TOOL, {"paths": []}),  # setup tool — no artifact
            (SEARCH_TOOL, {"results": []}),  # missing query_id
            (EXPORT_QUERY_TOOL, {"query_id": "q"}),  # missing download_url
            (RESOLVE_ENTITY_TOOL, {"candidates": []}),  # ambiguous resolve, no concrete entity
            (SEARCH_TOOL, "not-a-dict"),
        ],
    )
    def test_no_artifact_cases(self, tool, result):
        assert _artifact_for(tool, result) is None

    def test_unwraps_tool_return(self):
        result = ToolReturn(return_value={"query_id": "q-9", "total": 1})
        artifact = _artifact_for(SEARCH_TOOL, result)
        assert isinstance(artifact, QueryArtifact)
        assert artifact.query_id == "q-9"


class TestAggregateChart:
    """Deterministic Mermaid rendering of grouped aggregations (charts.aggregate_to_mermaid)."""

    # Core's `visualization` hint is one of pie/line/table; a "table" hint on a grouped result
    # still becomes a chart (default xychart bar) — that's the whole point of forcing the chart.
    GROUPED = {
        "visualization": "table",
        "results": [
            {"group_values": {"status": "active"}, "aggregations": {"count": 7}},
            {"group_values": {"status": "inactive"}, "aggregations": {"count": 2}},
        ],
    }

    def test_grouped_table_hint_renders_xychart_bar(self):
        # Raw Mermaid source (no fence — that's RenderedBlock.to_markdown's job).
        chart = aggregate_to_mermaid(self.GROUPED)
        assert chart.startswith("xychart-beta")
        assert 'x-axis ["active", "inactive"]' in chart
        assert "bar [7, 2]" in chart

    def test_visualization_value_selects_chart_type(self):
        # core echoes the LLM-chosen visualization_type into `visualization`: pie/line/bar.
        pie = aggregate_to_mermaid({**self.GROUPED, "visualization": "pie"})
        assert pie.startswith("pie title")
        assert '"active" : 7' in pie
        line = aggregate_to_mermaid({**self.GROUPED, "visualization": "line"})
        assert "line [7, 2]" in line
        bar = aggregate_to_mermaid({**self.GROUPED, "visualization": "bar"})
        assert bar.startswith("xychart-beta")
        assert "bar [7, 2]" in bar

    def test_scalar_aggregate_has_no_chart(self):
        # No group_values -> a single number -> nothing to chart.
        scalar = {"visualization": "table", "results": [{"group_values": {}, "aggregations": {"count": 7}}]}
        assert aggregate_to_mermaid(scalar) is None

    @pytest.mark.parametrize("payload", [{}, {"results": []}, {"results": "x"}, "nope"])
    def test_non_chartable_inputs_return_none(self, payload):
        assert aggregate_to_mermaid(payload) is None

    def test_rendered_block_to_markdown_fences_only_mermaid(self):
        assert RenderedBlock(type="mermaid", content="pie\n  x : 1").to_markdown() == "```mermaid\npie\n  x : 1\n```"
        assert RenderedBlock(type="markdown", content="| a |\n| - |").to_markdown() == "| a |\n| - |"

    def test_artifact_for_upgrades_grouped_aggregate_to_chart(self):
        payload = {"query_id": "q-agg", "total_results": 2, **self.GROUPED}
        artifact = _artifact_for(AGGREGATE_TOOL, payload)
        assert artifact.rendered_block is not None
        assert artifact.rendered_block.type == "mermaid"
        assert artifact.query_id == "q-agg"  # still a QueryArtifact (export keeps working)
        assert "xychart-beta" in artifact.rendered_block.content

    def test_artifact_for_scalar_aggregate_stays_query_artifact(self):
        payload = {
            "query_id": "q-1",
            "total_results": 7,
            "visualization": "table",
            "results": [{"group_values": {}, "aggregations": {"count": 7}}],
        }
        artifact = _artifact_for(AGGREGATE_TOOL, payload)
        assert isinstance(artifact, QueryArtifact)
        assert artifact.rendered_block is None


class TestSearchTable:
    """Deterministic Markdown table for search results (tables.search_to_markdown)."""

    RESULTS = {
        "query_id": "q-s",
        "returned": 2,
        "results": [
            {"entity_id": "11111111-aaaa", "entity_type": "SUBSCRIPTION", "title": "Node Access X", "score": 0.9},
            {"entity_id": "22222222-bbbb", "entity_type": "SUBSCRIPTION", "title": "IRB SP Y", "score": 0.8},
        ],
    }

    def test_renders_title_and_id_columns(self):
        md = search_to_markdown(self.RESULTS)
        assert "| Title | ID |" in md
        assert "| Node Access X | 11111111-aaaa |" in md
        assert "| IRB SP Y | 22222222-bbbb |" in md
        assert "Type" not in md  # single entity type -> no Type column

    def test_multiple_entity_types_add_type_column(self):
        rows = {
            "results": [
                {"entity_id": "1", "entity_type": "SUBSCRIPTION", "title": "A"},
                {"entity_id": "2", "entity_type": "PRODUCT", "title": "B"},
            ]
        }
        assert "| Type | Title | ID |" in search_to_markdown(rows)

    def test_row_cap_and_count(self):
        rows = {"results": [{"entity_id": str(i), "entity_type": "SUBSCRIPTION", "title": f"S{i}"} for i in range(15)]}
        md = search_to_markdown(rows)
        assert md.count("| S") == 10  # capped at 10 data rows
        assert "Showing 10 of 15" in md

    def test_no_rows_returns_none(self):
        assert search_to_markdown({"results": []}) is None
        assert search_to_markdown({"results": [1, 2]}) is None  # non-dict rows

    def test_artifact_for_upgrades_search_to_table(self):
        artifact = _artifact_for(SEARCH_TOOL, self.RESULTS)
        assert artifact.rendered_block is not None
        assert artifact.rendered_block.type == "markdown"
        assert artifact.query_id == "q-s"  # still a QueryArtifact (export keeps working)
        assert "| Title | ID |" in artifact.rendered_block.content


class TestDeterministicSearch:
    """The discover-before-filter guard (FilterPathGuard.before_tool_execute)."""

    CAP = FilterPathGuard()
    # A filter-consuming tool: its schema declares a `filters` property.
    SEARCH_DEF = ToolDefinition(
        name=SEARCH_TOOL,
        parameters_json_schema={"properties": {"entity_type": {}, "query_text": {}, "filters": {}}},
    )

    def _ctx(self, *tool_names: str):
        """A fake RunContext whose message history contains the given prior tool calls."""
        parts = [ToolCallPart(tool_name=name, args={}) for name in tool_names]
        return SimpleNamespace(messages=[SimpleNamespace(parts=parts)])

    async def _run(self, ctx, tool_def, args):
        return await self.CAP.before_tool_execute(
            ctx, call=ToolCallPart(tool_name=tool_def.name, args=args), tool_def=tool_def, args=args
        )

    async def test_blocks_filtered_search_without_discovery(self):
        with pytest.raises(ModelRetry, match=DISCOVER_FILTER_PATHS_TOOL):
            await self._run(self._ctx(), self.SEARCH_DEF, {"entity_type": "SUBSCRIPTION", "filters": {"a": 1}})

    async def test_allows_filtered_search_after_discovery(self):
        args = {"entity_type": "SUBSCRIPTION", "filters": {"a": 1}}
        assert await self._run(self._ctx(DISCOVER_FILTER_PATHS_TOOL), self.SEARCH_DEF, args) == args

    async def test_allows_search_without_filters(self):
        # No `filters` supplied -> no discovery prerequisite, even with no prior discovery call.
        args = {"entity_type": "SUBSCRIPTION", "query_text": "fiber"}
        assert await self._run(self._ctx(), self.SEARCH_DEF, args) == args

    async def test_ignores_tools_without_path_params(self):
        # A tool whose schema has no filters/group_by is never gated.
        export_def = ToolDefinition(name=EXPORT_QUERY_TOOL, parameters_json_schema={"properties": {"query_id": {}}})
        args = {"query_id": "q-1"}
        assert await self._run(self._ctx(), export_def, args) == args
