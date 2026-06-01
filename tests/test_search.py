"""Tests for run_search lenient/semantic-fallback helpers."""

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
from orchestrator.core.search.filters import EqualityFilter, FilterTree, PathFilter
from orchestrator.core.search.query.results import SearchResponse, SearchResult

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


def _filters() -> FilterTree:
    return FilterTree(
        children=[
            PathFilter(path="name", condition=EqualityFilter(op=FilterOp.EQ, value="acme"), value_kind=UIType.STRING)
        ],
        op=BooleanOperator.AND,
    )


def _state(user_input: str = "acme corp", with_filters: bool = True) -> SearchState:
    state = SearchState(user_input=user_input, run_id=RUN_ID)
    if with_filters:
        state.pending_filters = _filters()
    return state


class TestDescribeResults:
    @pytest.mark.parametrize(
        "count, fallback_used, expected",
        [
            pytest.param(5, False, "Found 5 matching SUBSCRIPTION", id="exact"),
            pytest.param(3, True, "No exact matches — showing 3 closest SUBSCRIPTION by similarity", id="fallback"),
        ],
    )
    def test_describe(self, count, fallback_used, expected):
        assert search_mod._describe_results(count, EntityType.SUBSCRIPTION, fallback_used) == expected


class TestExecuteSearchWithFallback:
    async def test_structured_results_skip_fallback(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            return _response([_result()], "structured"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()
        response, final_query, fallback_used = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert fallback_used is False
        assert len(response.results) == 1
        assert len(calls) == 1
        assert state.query_id == STRUCT_QID

    async def test_empty_structured_triggers_semantic_fallback(self, monkeypatch):
        async def fake(query, session, run_id):
            if query.filters is not None:
                return _response([], "structured"), RUN_ID, STRUCT_QID
            return _response([_result("s1")], "semantic"), RUN_ID, FALLBACK_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()
        response, final_query, fallback_used = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert fallback_used is True
        assert len(response.results) == 1
        assert response.metadata.search_type == "semantic"
        assert final_query.filters is None
        assert final_query.retriever == RetrieverType.SEMANTIC
        assert final_query.query_text == "acme corp"
        assert state.query_id == FALLBACK_QID

    async def test_empty_user_input_skips_fallback(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            return _response([], "structured"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state(user_input="", with_filters=True)
        response, final_query, fallback_used = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert fallback_used is False
        assert response.results == []
        assert len(calls) == 1

    async def test_semantic_valueerror_degrades_to_fuzzy(self, monkeypatch):
        seen_retrievers = []

        async def fake(query, session, run_id):
            seen_retrievers.append(query.retriever)
            if query.filters is not None:
                return _response([], "structured"), RUN_ID, STRUCT_QID
            if query.retriever == RetrieverType.SEMANTIC:
                raise ValueError("embedding unavailable")
            return _response([_result("f1")], "fuzzy"), RUN_ID, FALLBACK_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()
        response, final_query, fallback_used = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert fallback_used is True
        assert response.metadata.search_type == "fuzzy"
        assert final_query.retriever is None
        assert seen_retrievers == [None, RetrieverType.SEMANTIC, None]

    async def test_all_passes_empty_returns_structured(self, monkeypatch):
        calls = []

        async def fake(query, session, run_id):
            calls.append(query)
            search_type = "structured" if query.filters is not None else "semantic"
            return _response([], search_type), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        state = _state()
        response, final_query, fallback_used = await search_mod._execute_search_with_fallback(
            state, EntityType.SUBSCRIPTION, 10, object()
        )
        assert fallback_used is False
        assert response.results == []
        assert final_query.filters is not None
        assert len(calls) == 3
