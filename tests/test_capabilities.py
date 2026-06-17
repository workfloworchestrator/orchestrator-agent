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
from orchestrator_agent.capabilities.behavior import PluginCapability
from orchestrator_agent.capabilities.behavior.artifacts import data_artifact, export_artifact, query_artifact
from orchestrator_agent.capabilities.hooks import (
    DeferredToolGate,
    FilterPathGuard,
    trim_history,
)
from orchestrator_agent.capabilities.spec import PluginSpec
from orchestrator_agent.rendering.charts import aggregate_to_mermaid
from orchestrator_agent.rendering.tables import search_to_markdown
from orchestrator_agent.tool_names import (
    AGGREGATE_TOOL,
    DISCOVER_FILTER_PATHS_TOOL,
    EXPORT_QUERY_TOOL,
    RESOLVE_ENTITY_TOOL,
    SEARCH_TOOL,
)


def _artifact_for(tool: str, result):
    """Dispatch a tool result to its plugin's artifact builder — mirrors the behavior classes.

    The real per-tool mapping now lives in ``behavior/artifacts.py`` (one builder per artifact type);
    this helper keeps the mapping assertions below tool-keyed and exercises those builders directly.
    """
    payload = result.return_value if isinstance(result, ToolReturn) else result
    if tool in (SEARCH_TOOL, AGGREGATE_TOOL):
        return query_artifact(tool, payload)
    if tool == EXPORT_QUERY_TOOL:
        return export_artifact(tool, payload)
    if tool == RESOLVE_ENTITY_TOOL:
        return data_artifact(tool, payload)
    return None


class TestTrimHistory:
    def test_long_history_trimmed_to_window(self):
        msgs = [ModelRequest(parts=[UserPromptPart(content=str(i))]) for i in range(100)]
        trimmed = trim_history(msgs)
        assert len(trimmed) < len(msgs)
        # Most recent message is preserved.
        assert trimmed[-1] is msgs[-1]

    def test_leading_instructions_preserved(self):
        head = ModelRequest(parts=[UserPromptPart(content="start")], instructions="SYSTEM PROMPT")
        msgs = [head] + [ModelRequest(parts=[UserPromptPart(content=str(i))]) for i in range(100)]
        trimmed = trim_history(msgs)
        assert trimmed[0] is head


class TestPluginCapabilityBehavior:
    """The PluginCapability hook: ownership filtering + artifact attach via after_tool_execute."""

    # tools: are constant NAMES (as in frontmatter); the capability resolves them to live names.
    SPEC = PluginSpec(
        id="search", description="d", instructions="# Searching", defer_loading=False, tools=["SEARCH_TOOL"]
    )
    CAP = PluginCapability(SPEC, query_artifact)
    SEARCH_DEF = ToolDefinition(name=SEARCH_TOOL, parameters_json_schema={"properties": {}})
    OTHER_DEF = ToolDefinition(name=EXPORT_QUERY_TOOL, parameters_json_schema={"properties": {}})

    async def _run(self, tool_def, result):
        return await self.CAP.after_tool_execute(
            SimpleNamespace(),
            call=ToolCallPart(tool_name=tool_def.name, args={}),
            tool_def=tool_def,
            args={},
            result=result,
        )

    def test_instructions_and_identity_from_spec(self):
        assert self.CAP.id == "search"
        assert self.CAP.get_instructions() == "# Searching"
        assert self.CAP.defer_loading is False

    def test_owned_tools_resolved_to_live_names(self):
        # The ownership filter compares against live tool names; SEARCH_TOOL -> "search".
        assert self.CAP._tools == {SEARCH_TOOL}

    async def test_owned_tool_result_gets_artifact_metadata(self):
        out = await self._run(self.SEARCH_DEF, {"query_id": "q-1", "returned": 2, "results": []})
        assert isinstance(out, ToolReturn)
        assert isinstance(out.metadata, QueryArtifact)
        assert out.metadata.query_id == "q-1"

    async def test_unowned_tool_result_passes_through_untouched(self):
        # export is not in this plugin's tools — the hook must not touch its result.
        payload = {"query_id": "q-4", "download_path": "/x"}
        assert await self._run(self.OTHER_DEF, payload) is payload

    async def test_real_loaded_search_plugin_attaches_artifact(self):
        # Regression: frontmatter `tools:` are constant names; the capability must resolve them to
        # live names, else a real loaded plugin silently never attaches artifacts.
        from orchestrator_agent.capabilities.behavior import build_plugin_capability
        from orchestrator_agent.capabilities.loader import load_plugin_specs

        spec = {s.id: s for s in load_plugin_specs()}["search"]
        cap = build_plugin_capability(spec)  # selected by `artifact: query` declaration
        out = await cap.after_tool_execute(
            SimpleNamespace(),
            call=ToolCallPart(tool_name=SEARCH_TOOL, args={}),
            tool_def=ToolDefinition(name=SEARCH_TOOL, parameters_json_schema={"properties": {}}),
            args={},
            result={"query_id": "q-real", "returned": 1, "results": []},
        )
        assert isinstance(out, ToolReturn)
        assert isinstance(out.metadata, QueryArtifact)


class TestArtifactDeclaration:
    """build_plugin_capability selects an artifact builder by the frontmatter `artifact:` declaration."""

    def test_artifact_plugin_is_a_plugin_capability(self):
        from orchestrator_agent.capabilities.behavior import build_plugin_capability

        cap = build_plugin_capability(
            PluginSpec(id="x", description="d", instructions="i", defer_loading=False, artifact="query")
        )
        assert isinstance(cap, PluginCapability)

    def test_no_artifact_is_instructions_only(self):
        from pydantic_ai.capabilities import Capability

        from orchestrator_agent.capabilities.behavior import build_plugin_capability

        cap = build_plugin_capability(PluginSpec(id="z", description="d", instructions="i", defer_loading=False))
        assert isinstance(cap, Capability)
        assert not isinstance(cap, PluginCapability)

    def test_unknown_artifact_rejected_at_validation(self):
        # `artifact` is typed as ArtifactType — a bad value fails when the spec is built.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PluginSpec(id="z", description="d", instructions="i", defer_loading=False, artifact="bogus")

    @pytest.mark.parametrize(
        "artifact, tool, payload, expected",
        [
            ("query", SEARCH_TOOL, {"query_id": "q", "returned": 1, "results": []}, QueryArtifact),
            ("data", RESOLVE_ENTITY_TOOL, {"entity_id": "e", "entity_type": "SUBSCRIPTION"}, DataArtifact),
            ("export", EXPORT_QUERY_TOOL, {"query_id": "q", "download_path": "/x"}, ExportArtifact),
        ],
    )
    async def test_declared_binding_attaches_right_artifact_end_to_end(self, artifact, tool, payload, expected):
        # Drives the real artifact: -> class -> builder path (not the test's _artifact_for helper),
        # so a mis-bound ARTIFACT_BEHAVIORS or broken artifact() method would fail here.
        from orchestrator_agent.capabilities.behavior import build_plugin_capability

        tool_const = {
            SEARCH_TOOL: "SEARCH_TOOL",
            RESOLVE_ENTITY_TOOL: "RESOLVE_ENTITY_TOOL",
            EXPORT_QUERY_TOOL: "EXPORT_QUERY_TOOL",
        }[tool]
        cap = build_plugin_capability(
            PluginSpec(
                id="p", description="d", instructions="i", defer_loading=False, tools=[tool_const], artifact=artifact
            )
        )
        out = await cap.after_tool_execute(
            SimpleNamespace(),
            call=ToolCallPart(tool_name=tool, args={}),
            tool_def=ToolDefinition(name=tool, parameters_json_schema={"properties": {}}),
            args={},
            result=payload,
        )
        assert isinstance(out, ToolReturn)
        assert isinstance(out.metadata, expected)


class TestDeferredToolGate:
    """The gate hides a deferred plugin's owned tools until its capability is loaded."""

    DEFERRED_SEARCH = PluginSpec(
        id="search", description="d", instructions="i", tools=["SEARCH_TOOL"], defer_loading=True
    )
    ALWAYS_AGG = PluginSpec(
        id="aggregate", description="d", instructions="i", defer_loading=False, tools=["AGGREGATE_TOOL"]
    )
    GATE = DeferredToolGate([DEFERRED_SEARCH, ALWAYS_AGG])
    DEFS = [ToolDefinition(name=n) for n in (SEARCH_TOOL, AGGREGATE_TOOL, DISCOVER_FILTER_PATHS_TOOL)]

    async def _names(self, loaded):
        ctx = SimpleNamespace(loaded_capability_ids=set(loaded))
        return {t.name for t in await self.GATE.prepare_tools(ctx, list(self.DEFS))}

    async def test_deferred_tool_hidden_until_loaded(self):
        assert SEARCH_TOOL not in await self._names(loaded=[])
        assert SEARCH_TOOL in await self._names(loaded=["search"])

    async def test_always_on_and_unowned_tools_never_hidden(self):
        names = await self._names(loaded=[])
        assert AGGREGATE_TOOL in names  # always-on plugin's tool
        assert DISCOVER_FILTER_PATHS_TOOL in names  # owned by no plugin (base)

    async def test_no_deferred_plugins_is_passthrough(self):
        gate = DeferredToolGate([self.ALWAYS_AGG])
        ctx = SimpleNamespace(loaded_capability_ids=set())
        out = await gate.prepare_tools(ctx, list(self.DEFS))
        assert out == self.DEFS

    async def test_two_deferred_plugins_reveal_independently(self):
        deferred_agg = PluginSpec(
            id="aggregate", description="d", instructions="i", tools=["AGGREGATE_TOOL"], defer_loading=True
        )
        gate = DeferredToolGate([self.DEFERRED_SEARCH, deferred_agg])
        ctx = SimpleNamespace(loaded_capability_ids={"search"})  # only search loaded
        names = {t.name for t in await gate.prepare_tools(ctx, list(self.DEFS))}
        assert SEARCH_TOOL in names  # loaded
        assert AGGREGATE_TOOL not in names  # still deferred-unloaded

    async def test_tool_shared_with_always_on_owner_is_never_hidden(self):
        # A tool owned by both a deferred and an always-on plugin stays visible (always-on wins).
        deferred_dup = PluginSpec(
            id="search", description="d", instructions="i", tools=["AGGREGATE_TOOL"], defer_loading=True
        )
        gate = DeferredToolGate([deferred_dup, self.ALWAYS_AGG])  # both own AGGREGATE_TOOL
        ctx = SimpleNamespace(loaded_capability_ids=set())
        names = {t.name for t in await gate.prepare_tools(ctx, list(self.DEFS))}
        assert AGGREGATE_TOOL in names


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
