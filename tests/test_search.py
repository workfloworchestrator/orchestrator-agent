"""Tests for run_search broadening-fallback helpers."""

from __future__ import annotations

import os
from uuid import uuid4

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from orchestrator.core.search.core.types import (
    BooleanOperator,
    EntityType,
    FilterOp,
    RetrieverType,
    SearchMetadata,
    UIType,
)
from orchestrator.core.search.filters import (
    EqualityFilter,
    FilterTree,
    NumericValueFilter,
    PathFilter,
    StringFilter,
)
from orchestrator.core.search.query.results import SearchResponse, SearchResult

from orchestrator_agent.settings import SearchEffort
from orchestrator_agent.state import SearchState
from orchestrator_agent.tools import search as search_mod

RUN_ID = uuid4()
STRUCT_QID = uuid4()
FALLBACK_QID = uuid4()


def _result(entity_id: str = "e1") -> SearchResult:
    return SearchResult(
        entity_id=entity_id,
        entity_type=EntityType.SUBSCRIPTION,
        entity_title="Title",
        score=0.9,
    )


def _response(results: list[SearchResult], search_type: str) -> SearchResponse:
    return SearchResponse(results=results, metadata=SearchMetadata(search_type=search_type, description="x"))


def _eq(path: str, value: str) -> PathFilter:
    return PathFilter(path=path, condition=EqualityFilter(op=FilterOp.EQ, value=value), value_kind=UIType.STRING)


def _like(path: str, value: str) -> PathFilter:
    return PathFilter(path=path, condition=StringFilter(op=FilterOp.LIKE, value=value), value_kind=UIType.STRING)


def _num_gt(path: str, value: int) -> PathFilter:
    return PathFilter(path=path, condition=NumericValueFilter(op=FilterOp.GT, value=value), value_kind=UIType.NUMBER)


def _tree(*leaves: PathFilter) -> FilterTree:
    return FilterTree(op=BooleanOperator.AND, children=list(leaves))


def _filters() -> FilterTree:
    """A single exact (eq) leaf — has nothing relaxable, so it skips the relaxed rung."""
    return _tree(_eq("name", "acme"))


def _state(user_input: str = "acme corp", with_filters: bool = True) -> SearchState:
    state = SearchState(user_input=user_input, run_id=RUN_ID)
    if with_filters:
        state.pending_filters = _filters()
    return state


def _has_like(filters: FilterTree | None) -> bool:
    return filters is not None and any(isinstance(leaf.condition, StringFilter) for leaf in filters.get_all_leaves())


class TestHighSignalFilters:
    def test_drops_like_keeps_exact(self):
        tree = _tree(
            _eq("subscription.customer_id", "uuid-x"),
            _eq("subscription.status", "active"),
            _like("subscription.product.name", "%bgp%"),
        )
        reduced = search_mod._high_signal_filters(tree)
        assert reduced is not None
        assert reduced.get_all_paths() == {"subscription.customer_id", "subscription.status"}

    def test_keeps_range_drops_like(self):
        tree = _tree(_num_gt("subscription.bandwidth", 100), _like("subscription.product.name", "%bgp%"))
        reduced = search_mod._high_signal_filters(tree)
        assert reduced is not None
        assert reduced.get_all_paths() == {"subscription.bandwidth"}

    @pytest.mark.parametrize(
        "tree",
        [
            pytest.param(None, id="none-input"),
            pytest.param(_tree(_eq("subscription.status", "active")), id="nothing-relaxable"),
            pytest.param(_tree(_like("subscription.product.name", "%bgp%")), id="nothing-high-signal"),
        ],
    )
    def test_returns_none_when_relaxation_is_not_useful(self, tree):
        assert search_mod._high_signal_filters(tree) is None


class TestDescribeResults:
    @pytest.mark.parametrize(
        "count, mode, expected",
        [
            pytest.param(5, search_mod.MatchMode.EXACT, "Found 5 matching SUBSCRIPTION", id="exact"),
            pytest.param(
                4,
                search_mod.MatchMode.RELAXED,
                "No exact match on all criteria — showing 4 SUBSCRIPTION matching the key filters",
                id="relaxed",
            ),
            pytest.param(
                3,
                search_mod.MatchMode.SIMILARITY,
                "No exact matches — showing 3 closest SUBSCRIPTION by similarity",
                id="similarity",
            ),
        ],
    )
    def test_describe(self, count, mode, expected):
        assert search_mod._describe_results(count, EntityType.SUBSCRIPTION, mode) == expected


class TestExecuteSearchWithFallback:
    async def test_structured_results_skip_fallback(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            return _response([_result()], "structured"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()
        response, final_query, mode = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert mode == search_mod.MatchMode.EXACT
        assert len(response.results) == 1
        assert len(calls) == 1
        assert state.query_id == STRUCT_QID

    async def test_relaxed_pass_preserves_high_signal_filters(self, monkeypatch):
        """An over-constraining `like` filter is relaxed while exact id/status filters are kept."""
        tree = _tree(
            _eq("subscription.customer_id", "uuid-x"),
            _eq("subscription.status", "active"),
            _like("subscription.product.name", "%bgp%"),
        )

        async def fake(query, session, run_id):
            if _has_like(query.filters):
                return _response([], "structured"), RUN_ID, STRUCT_QID
            if query.filters is not None:
                return _response([_result("r1")], "structured"), RUN_ID, FALLBACK_QID
            return _response([_result("s1")], "semantic"), RUN_ID, uuid4()

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = SearchState(user_input="uva bgp", run_id=RUN_ID)
        state.pending_filters = tree
        response, final_query, mode = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object(), effort=SearchEffort.MEDIUM
        )
        assert mode == search_mod.MatchMode.RELAXED
        assert final_query.filters is not None
        assert final_query.filters.get_all_paths() == {"subscription.customer_id", "subscription.status"}
        assert final_query.retriever is None
        assert state.query_id == FALLBACK_QID

    async def test_filterless_passes_try_hybrid_before_semantic(self, monkeypatch):
        seen = []

        async def fake(query, session, run_id):
            seen.append(query.retriever)
            return _response([], "structured" if query.filters is not None else "semantic"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", True)
        state = _state()  # single eq leaf -> no relaxed rung
        await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object(), effort=SearchEffort.HIGH
        )
        assert seen == [None, RetrieverType.HYBRID, RetrieverType.SEMANTIC]

    async def test_filterless_passes_degrade_to_fuzzy_without_embeddings(self, monkeypatch):
        seen = []

        async def fake(query, session, run_id):
            seen.append(query.retriever)
            if query.filters is not None:
                return _response([], "structured"), RUN_ID, STRUCT_QID
            return _response([_result("f1")], "fuzzy"), RUN_ID, FALLBACK_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", False)
        state = _state()
        response, final_query, mode = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object(), effort=SearchEffort.HIGH
        )
        assert mode == search_mod.MatchMode.SIMILARITY
        assert final_query.retriever == RetrieverType.FUZZY
        assert seen[:2] == [None, RetrieverType.FUZZY]

    async def test_valueerror_skips_to_next_filterless_pass(self, monkeypatch):
        seen = []

        async def fake(query, session, run_id):
            seen.append(query.retriever)
            if query.filters is not None:
                return _response([], "structured"), RUN_ID, STRUCT_QID
            if query.retriever == RetrieverType.HYBRID:
                raise ValueError("embedding unavailable")
            return _response([_result("s1")], "semantic"), RUN_ID, FALLBACK_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", True)
        state = _state()
        response, final_query, mode = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object(), effort=SearchEffort.HIGH
        )
        assert mode == search_mod.MatchMode.SIMILARITY
        assert final_query.retriever == RetrieverType.SEMANTIC
        assert seen == [None, RetrieverType.HYBRID, RetrieverType.SEMANTIC]

    async def test_empty_user_input_skips_fallback(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            return _response([], "structured"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state(user_input="", with_filters=True)
        response, final_query, mode = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert mode == search_mod.MatchMode.EXACT
        assert response.results == []
        assert len(calls) == 1

    async def test_all_passes_empty_returns_structured(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            search_type = "structured" if query.filters is not None else "semantic"
            return _response([], search_type), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()  # single eq leaf -> no relaxed rung; HIGH tries both filterless rungs
        response, final_query, mode = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object(), effort=SearchEffort.HIGH
        )
        assert mode == search_mod.MatchMode.EXACT
        assert response.results == []
        assert final_query.filters is not None
        assert len(calls) == 3

    @pytest.mark.parametrize(
        "effort, expected_calls",
        [
            pytest.param(SearchEffort.LOW, 1, id="low-no-fallback"),
            pytest.param(SearchEffort.MEDIUM, 2, id="medium-one-pass"),
            pytest.param(SearchEffort.HIGH, 3, id="high-two-filterless-passes"),
        ],
    )
    async def test_effort_controls_passes_without_relaxable_filter(self, monkeypatch, effort, expected_calls):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            search_type = "structured" if query.filters is not None else "semantic"
            return _response([], search_type), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()  # single eq leaf -> ladder is [hybrid, semantic]
        await search_mod._execute_search_with_fallback(state, EntityType.SUBSCRIPTION, 10, object(), effort=effort)
        assert len(calls) == expected_calls

    @pytest.mark.parametrize(
        "effort, expected_calls",
        [
            pytest.param(SearchEffort.LOW, 1, id="low-no-fallback"),
            pytest.param(SearchEffort.MEDIUM, 2, id="medium-relaxed-only"),
            pytest.param(SearchEffort.HIGH, 4, id="high-relaxed-plus-filterless"),
        ],
    )
    async def test_effort_controls_passes_with_relaxable_filter(self, monkeypatch, effort, expected_calls):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            search_type = "structured" if query.filters is not None else "semantic"
            return _response([], search_type), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = SearchState(user_input="uva bgp", run_id=RUN_ID)
        state.pending_filters = _tree(_eq("subscription.status", "active"), _like("subscription.product.name", "%bgp%"))
        await search_mod._execute_search_with_fallback(state, EntityType.SUBSCRIPTION, 10, object(), effort=effort)
        assert len(calls) == expected_calls

    async def test_effort_defaults_to_configured_setting(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            search_type = "structured" if query.filters is not None else "semantic"
            return _response([], search_type), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        monkeypatch.setattr(search_mod.agent_settings, "AGENT_SEARCH_EFFORT", SearchEffort.LOW)
        state = _state()
        await search_mod._execute_search_with_fallback(state, EntityType.SUBSCRIPTION, 10, object())
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "embeddings_enabled, requested, expected",
        [
            pytest.param(True, RetrieverType.HYBRID, RetrieverType.HYBRID, id="hybrid-on"),
            pytest.param(False, RetrieverType.HYBRID, RetrieverType.FUZZY, id="hybrid-off-degrades"),
            pytest.param(True, None, None, id="auto-passthrough"),
        ],
    )
    async def test_primary_pass_uses_effective_retriever(self, monkeypatch, embeddings_enabled, requested, expected):
        captured = []

        async def fake(query, session, run_id):
            captured.append(query.retriever)
            return _response([_result()], "structured"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", embeddings_enabled)
        state = _state()
        await search_mod._execute_search_with_fallback(state, EntityType.SUBSCRIPTION, 10, object(), requested)
        assert captured[0] == expected


class TestRunSearchArtifact:
    @pytest.mark.parametrize(
        "mode, search_type, expected_description",
        [
            pytest.param(search_mod.MatchMode.EXACT, "structured", "Found 1 matching SUBSCRIPTION", id="exact"),
            pytest.param(
                search_mod.MatchMode.RELAXED,
                "structured",
                "No exact match on all criteria — showing 1 SUBSCRIPTION matching the key filters",
                id="relaxed",
            ),
            pytest.param(
                search_mod.MatchMode.SIMILARITY,
                "semantic",
                "No exact matches — showing 1 closest SUBSCRIPTION by similarity",
                id="similarity",
            ),
        ],
    )
    async def test_run_search_propagates_match_mode(self, monkeypatch, mode, search_type, expected_description):
        from types import SimpleNamespace

        from orchestrator.core.search.query.queries import SelectQuery

        response = _response([_result("e1")], search_type)
        final_query = SelectQuery(entity_type=EntityType.SUBSCRIPTION, query_text="acme corp", filters=None, limit=10)

        async def fake_execute(state, entity_type, limit, session, retriever=None):
            state.query_id = STRUCT_QID
            return response, final_query, mode

        monkeypatch.setattr(search_mod, "_execute_search_with_fallback", fake_execute)
        monkeypatch.setattr(search_mod, "db", SimpleNamespace(session=object()))

        state = SearchState(user_input="acme corp", run_id=RUN_ID)
        state.query_id = STRUCT_QID
        state.memory.start_turn("acme corp")
        state.memory.start_step("Search")
        ctx = SimpleNamespace(deps=SimpleNamespace(state=state))

        tool_return = await search_mod.run_search(ctx, EntityType.SUBSCRIPTION, 10)

        artifact = tool_return.metadata
        assert artifact.search_type == search_type
        assert artifact.description == expected_description
        assert artifact.total_results == 1

        recorded = state.memory.current_turn.current_step.tool_steps[-1]
        assert recorded.context["match_mode"] == mode.value
        assert recorded.context["search_type"] == search_type


async def test_run_search_forwards_retriever(monkeypatch):
    from types import SimpleNamespace

    from orchestrator.core.search.query.queries import SelectQuery

    captured = {}

    async def fake_execute(state, entity_type, limit, session, retriever):
        captured["retriever"] = retriever
        state.query_id = STRUCT_QID
        final_query = SelectQuery(entity_type=EntityType.SUBSCRIPTION, query_text="acme", filters=None, limit=10)
        return _response([_result("e1")], "structured"), final_query, search_mod.MatchMode.EXACT

    monkeypatch.setattr(search_mod, "_execute_search_with_fallback", fake_execute)
    monkeypatch.setattr(search_mod, "db", SimpleNamespace(session=object()))

    state = SearchState(user_input="acme", run_id=RUN_ID)
    state.query_id = STRUCT_QID
    state.memory.start_turn("acme")
    state.memory.start_step("Search")
    ctx = SimpleNamespace(deps=SimpleNamespace(state=state))

    await search_mod.run_search(ctx, EntityType.SUBSCRIPTION, 10, RetrieverType.HYBRID)
    assert captured["retriever"] == RetrieverType.HYBRID


async def test_run_search_limit_defaults_to_setting_and_honors_override(monkeypatch):
    from types import SimpleNamespace

    from orchestrator.core.search.query.queries import SelectQuery

    captured = {}

    async def fake_execute(state, entity_type, limit, session, retriever=None):
        captured["limit"] = limit
        state.query_id = STRUCT_QID
        final_query = SelectQuery(entity_type=EntityType.SUBSCRIPTION, query_text="acme", filters=None, limit=limit)
        return _response([_result("e1")], "structured"), final_query, search_mod.MatchMode.EXACT

    monkeypatch.setattr(search_mod, "_execute_search_with_fallback", fake_execute)
    monkeypatch.setattr(search_mod, "db", SimpleNamespace(session=object()))
    monkeypatch.setattr(search_mod.agent_settings, "SEARCH_RESULT_LIMIT", 25)

    state = SearchState(user_input="acme", run_id=RUN_ID)
    state.query_id = STRUCT_QID
    state.memory.start_turn("acme")
    state.memory.start_step("Search")
    ctx = SimpleNamespace(deps=SimpleNamespace(state=state))

    await search_mod.run_search(ctx, EntityType.SUBSCRIPTION)
    assert captured["limit"] == 25

    await search_mod.run_search(ctx, EntityType.SUBSCRIPTION, 5)
    assert captured["limit"] == 5


class TestEffectiveRetriever:
    @pytest.mark.parametrize(
        "embeddings_enabled, requested, expected",
        [
            pytest.param(True, RetrieverType.HYBRID, RetrieverType.HYBRID, id="on-hybrid"),
            pytest.param(True, RetrieverType.SEMANTIC, RetrieverType.SEMANTIC, id="on-semantic"),
            pytest.param(True, RetrieverType.FUZZY, RetrieverType.FUZZY, id="on-fuzzy"),
            pytest.param(True, None, None, id="on-auto"),
            pytest.param(False, RetrieverType.HYBRID, RetrieverType.FUZZY, id="off-hybrid-degrades"),
            pytest.param(False, RetrieverType.SEMANTIC, RetrieverType.FUZZY, id="off-semantic-degrades"),
            pytest.param(False, RetrieverType.FUZZY, RetrieverType.FUZZY, id="off-fuzzy"),
            pytest.param(False, None, None, id="off-auto"),
        ],
    )
    def test_effective(self, monkeypatch, embeddings_enabled, requested, expected):
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", embeddings_enabled)
        assert search_mod._effective_retriever(requested) == expected
