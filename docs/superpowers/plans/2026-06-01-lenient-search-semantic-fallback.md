# Lenient Filtering and Semantic-Search Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent's searches far less likely to return nothing — guide the LLM toward lenient filter operators and automatically fall back to a filterless semantic search when a structured search returns zero results.

**Architecture:** Two coordinated changes. (1) Prompt guidance in `prompts.py` teaches the agent to prefer `LIKE`/range operators over `EQ`. (2) A deterministic safety-net in `tools/search.py`: `run_search` runs the structured pass; on exactly zero results (and non-empty user input) it re-runs as a filterless semantic pass (degrading to fuzzy if embeddings are unavailable). Results are labeled exact vs. approximate via the tool description and a new `search_type` field on the streamed artifact. We reuse `orchestrator-core`'s existing `SelectQuery.retriever` and `SemanticRetriever` — no new retrieval engine.

**Tech Stack:** Python 3.11–3.13, `uv`, `pydantic-ai`, `orchestrator-core[search]`, `pytest`/`pytest-asyncio` (asyncio_mode=auto), ruff + black (line-length 120), mypy (strict, `src/` only).

**Project Python style rules (apply to all code written here):** prefer comprehensions over imperative loops; avoid `break`/`continue` (use early return in helpers); prefer `match/case` over `isinstance` chains; for tests use `@pytest.mark.parametrize` (with `pytest.param(..., id=...)`) instead of near-duplicate test functions.

**Commit convention (from project memory):** no `Co-Authored-By` lines; prefix commit commands with `export PATH="$HOME/.local/bin:$PATH"` so pre-commit hooks find `uv`.

---

## File Structure

- **Modify** `src/orchestrator_agent/artifacts.py` — add optional `search_type` field to `QueryArtifact` so the AG-UI artifact can badge results as exact/approximate without a REST round-trip.
- **Modify** `src/orchestrator_agent/prompts.py` — add operator-leniency guidance to shared `FILTERING_RULES`; add a fallback note to the search prompt only.
- **Modify** `src/orchestrator_agent/skills.py` — update the `SEARCH` skill `description` and `tags` to advertise lenient + semantic-fallback behavior.
- **Modify** `src/orchestrator_agent/tools/search.py` — add fallback helpers and rewrite `run_search` to use them. This is the core change.
- **Create** `tests/test_search.py` — unit tests for the fallback helpers and the description builder.
- **Modify** `tests/test_artifacts.py`, `tests/test_prompts.py`, `tests/test_skills.py` — cover the new field, prompt text, and skill metadata.

---

## Task 1: Add `search_type` field to `QueryArtifact`

**Files:**
- Modify: `src/orchestrator_agent/artifacts.py:36-44`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write the failing tests**

Add these two methods to the existing `class TestQueryArtifact:` in `tests/test_artifacts.py`:

```python
    def test_search_type_defaults_to_empty(self):
        a = QueryArtifact(description="test", query_id="q1", total_results=0)
        assert a.search_type == ""

    def test_search_type_can_be_set(self):
        a = QueryArtifact(description="test", query_id="q1", total_results=3, search_type="semantic")
        assert a.search_type == "semantic"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL — `TypeError`/`ValidationError` (unexpected keyword `search_type`) or `AttributeError: search_type`.

- [ ] **Step 3: Add the field**

In `src/orchestrator_agent/artifacts.py`, change `QueryArtifact` to add the field after `visualization_type`:

```python
class QueryArtifact(ToolArtifact):
    """Lightweight reference returned by query tools.

    Client fetches full results via GET /queries/{query_id}/results.
    """

    query_id: str
    total_results: int
    visualization_type: VisualizationType = Field(default_factory=VisualizationType)
    search_type: str = ""
```

(The default `""` keeps existing `QueryArtifact(...)` call sites — including `tools/aggregation.py:110` — working unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/artifacts.py tests/test_artifacts.py
git commit -m "Add search_type field to QueryArtifact"
```

---

## Task 2: Add lenient-operator guidance and the search fallback note to prompts

**Files:**
- Modify: `src/orchestrator_agent/prompts.py:24-30` (FILTERING_RULES) and `:44-70` (search prompt template)
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prompts.py`. Extend `class TestGetSearchExecutionPrompt:` with two methods, and `class TestGetAggregationExecutionPrompt:` with one:

```python
    def test_includes_lenient_operator_guidance(self):
        prompt = get_search_execution_prompt(_make_state())
        assert "PREFER LENIENT OPERATORS" in prompt
        assert "like" in prompt
        assert "between" in prompt

    def test_includes_semantic_fallback_note(self):
        prompt = get_search_execution_prompt(_make_state())
        assert "automatically retries with a broader semantic search" in prompt
```

```python
    def test_aggregation_has_no_semantic_fallback_note(self):
        prompt = get_aggregation_execution_prompt(_make_state())
        assert "automatically retries with a broader semantic search" not in prompt
        # but it DOES share the lenient operator guidance
        assert "PREFER LENIENT OPERATORS" in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL — `assert "PREFER LENIENT OPERATORS" in prompt` and the fallback-note assertions fail (text not present yet).

- [ ] **Step 3: Add operator guidance to `FILTERING_RULES`**

In `src/orchestrator_agent/prompts.py`, replace the `FILTERING_RULES` definition (lines 24-30) with:

```python
FILTERING_RULES = f"""### Filtering Rules (if query requires filters)
- **MANDATORY FIRST STEP**: You MUST call `{tools.discover_filter_paths.__name__}` BEFORE calling `{tools.set_filter_tree.__name__}`. Never skip this — filter paths are database-specific and cannot be guessed.
- Pass simple field names to discovery (e.g. "status", "id", "start_date") — not dotted paths like "subscription.status"
- **USE EXACT PATHS**: Only use paths returned by `{tools.discover_filter_paths.__name__}`. Do not modify or invent paths.
- **MATCH OPERATORS**: Only use operators compatible with the field type as confirmed by `{tools.get_valid_operators.__name__}`
- **PREFER LENIENT OPERATORS** — choose the broadest operator that still captures the user's intent:
  - Text, names, titles, descriptions, or partial values → use `like` (substring match, e.g. `%acme%`), NOT `eq`. Reserve `eq` for exact identifiers (UUIDs), enum/status values, and booleans.
  - Dates and numbers ("in 2025", "after X", "between X and Y", "more than 100") → use range operators `between`/`gt`/`gte`/`lt`/`lte`, NOT `eq`.
  - Avoid `eq` on human-typed text: an over-strict filter that matches nothing is worse than a broad one.
- Temporal constraints like "in 2025", "between X and Y" require filters on datetime fields
- If a discovered path does not match the user's intent, try alternative field names in a new discovery call"""
```

- [ ] **Step 4: Add the fallback note to the search prompt only**

In `src/orchestrator_agent/prompts.py`, in `get_search_execution_prompt`, the template currently ends the rules block with `{FILTERING_RULES}` followed by `---` and `{context}`. Insert a note between `{FILTERING_RULES}` and the `---` separator so that section reads:

```python
        {FILTERING_RULES}

        **Note:** If a search returns no results, the system automatically retries with a broader semantic search (filters dropped) and shows the closest matches. Don't over-constrain your filters. If the result description says matches are approximate, briefly tell the user the results are the closest available rather than exact matches.

        ---

        {context}
```

Leave `get_aggregation_execution_prompt` unchanged — it already interpolates `{FILTERING_RULES}` (so it gets the operator guidance) but must NOT get the fallback note.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones).

- [ ] **Step 6: Commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/prompts.py tests/test_prompts.py
git commit -m "Guide agent toward lenient filter operators and note semantic fallback"
```

---

## Task 3: Advertise lenient + fallback behavior in the SEARCH skill

**Files:**
- Modify: `src/orchestrator_agent/skills.py:60-68`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestSkillsRegistry:` in `tests/test_skills.py`:

```python
    def test_search_skill_advertises_semantic_fallback(self):
        skill = SKILLS[TaskAction.SEARCH]
        assert "semantic" in skill.description.lower()
        assert "semantic" in skill.tags
        assert "fuzzy" in skill.tags
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_skills.py -v`
Expected: FAIL — `assert "semantic" in skill.description.lower()` fails (current description has no "semantic").

- [ ] **Step 3: Update the SEARCH skill entry**

In `src/orchestrator_agent/skills.py`, update the `TaskAction.SEARCH` entry's `description` and `tags`:

```python
    TaskAction.SEARCH: Skill(
        action=TaskAction.SEARCH,
        name="Search",
        description=(
            "Find subscriptions, products, workflows, processes. Uses lenient filters "
            "(partial text and ranges) and falls back to semantic similarity search when "
            "no exact matches are found."
        ),
        tags=["search", "query", "fuzzy", "semantic"],
        toolsets=[filter_building_toolset, search_execution_toolset],
        get_prompt=get_search_execution_prompt,
        memory_scope=MemoryScope.LIGHTWEIGHT,
    ),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_skills.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/skills.py tests/test_skills.py
git commit -m "Advertise lenient + semantic-fallback behavior in SEARCH skill"
```

---

## Task 4: Add fallback helpers and rewrite `run_search`

This is the core change. The fallback logic lives in pure, RunContext-free helpers so it is directly unit-testable; `run_search` becomes a thin wrapper that orchestrates them and builds the response/artifact.

**Files:**
- Modify: `src/orchestrator_agent/tools/search.py` (whole file rewrite below)
- Create: `tests/test_search.py`

### 4a. Write the failing tests

- [ ] **Step 1: Create `tests/test_search.py` with the helper tests**

Create `tests/test_search.py`:

```python
"""Tests for run_search lenient/semantic-fallback helpers."""

from __future__ import annotations

import os
from uuid import uuid4

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from orchestrator.core.search.core.types import EntityType, RetrieverType, SearchMetadata, UIType
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
            PathFilter(path="name", condition=EqualityFilter(op="eq", value="acme"), value_kind=UIType.STRING)
        ],
        op="AND",
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
            pytest.param(
                3, True, "No exact matches — showing 3 closest SUBSCRIPTION by similarity", id="fallback"
            ),
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL — `AttributeError: module 'orchestrator_agent.tools.search' has no attribute '_describe_results'` (and `_execute_search_with_fallback`).

### 4b. Implement the helpers and rewrite `run_search`

- [ ] **Step 3: Rewrite `src/orchestrator_agent/tools/search.py`**

Replace the entire body below the license header (lines 14 to end) with:

```python
from typing import cast
from uuid import UUID

import structlog
from orchestrator.core.db import db
from orchestrator.core.db.database import WrappedSession
from orchestrator.core.search.core.types import EntityType, RetrieverType
from orchestrator.core.search.query.queries import Query, SelectQuery
from orchestrator.core.search.query.results import (
    QueryResultsResponse,
    ResultRow,
    SearchResponse,
    VisualizationType,
)
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.handlers import execute_search_with_persistence
from orchestrator_agent.memory import ToolStep
from orchestrator_agent.state import SearchState
from orchestrator_agent.tools.filters import ensure_query_initialized

logger = structlog.get_logger(__name__)

search_execution_toolset: FunctionToolset[StateDeps[SearchState]] = FunctionToolset(max_retries=2)


def _describe_results(count: int, entity_type: EntityType, fallback_used: bool) -> str:
    """Summarize results, distinguishing exact filter matches from approximate ones."""
    label = entity_type.value
    if fallback_used:
        return f"No exact matches — showing {count} closest {label} by similarity"
    return f"Found {count} matching {label}"


async def _attempt_semantic_query(
    retriever: RetrieverType | None,
    *,
    entity_type: EntityType,
    query_text: str,
    limit: int,
    run_id: UUID | None,
    session: WrappedSession,
) -> tuple[SearchResponse, UUID, UUID, SelectQuery] | None:
    """Run one filterless ranking pass; return it only if it produced rows.

    A ValueError means the embedding was unavailable for an explicit retriever
    override — treat it as "no help" so the caller can try the next strategy.
    """
    query = SelectQuery(
        entity_type=entity_type,
        query_text=query_text,
        filters=None,
        retriever=retriever,
        limit=limit,
    )
    try:
        response, new_run_id, query_id = await execute_search_with_persistence(query, session, run_id)
    except ValueError:
        return None
    return (response, new_run_id, query_id, query) if response.results else None


async def _run_semantic_fallback(
    *,
    entity_type: EntityType,
    query_text: str,
    limit: int,
    run_id: UUID | None,
    session: WrappedSession,
) -> tuple[SearchResponse, UUID, UUID, SelectQuery] | None:
    """Last-resort search: filterless semantic ranking, degrading to auto-routed fuzzy.

    SemanticRetriever applies no similarity threshold, so with filters dropped it
    returns the closest N embedded entities and is effectively guaranteed non-empty.
    If the explicit semantic pass cannot embed the query (ValueError), the
    auto-routed pass (retriever=None) degrades to fuzzy in orchestrator-core.
    """
    return await _attempt_semantic_query(
        RetrieverType.SEMANTIC,
        entity_type=entity_type,
        query_text=query_text,
        limit=limit,
        run_id=run_id,
        session=session,
    ) or await _attempt_semantic_query(
        None,
        entity_type=entity_type,
        query_text=query_text,
        limit=limit,
        run_id=run_id,
        session=session,
    )


async def _execute_search_with_fallback(
    state: SearchState,
    entity_type: EntityType,
    limit: int,
    session: WrappedSession,
) -> tuple[SearchResponse, SelectQuery, bool]:
    """Run the structured pass, then a semantic fallback when it returns zero rows.

    Updates state.run_id/query_id to the query that produced the returned rows.
    Returns (response, final_query, fallback_used).
    """
    ensure_query_initialized(state, entity_type)
    query = cast(SelectQuery, cast(Query, state.query).model_copy(update={"limit": limit}))
    response, run_id, query_id = await execute_search_with_persistence(query, session, state.run_id)
    state.run_id = run_id
    state.query_id = query_id

    if response.results or not state.user_input:
        return response, query, False

    fallback = await _run_semantic_fallback(
        entity_type=entity_type,
        query_text=state.user_input,
        limit=limit,
        run_id=state.run_id,
        session=session,
    )
    if fallback is None:
        return response, query, False

    fb_response, fb_run_id, fb_query_id, fb_query = fallback
    state.run_id = fb_run_id
    state.query_id = fb_query_id
    return fb_response, fb_query, True


@search_execution_toolset.tool
async def run_search(
    ctx: RunContext[StateDeps[SearchState]],
    entity_type: EntityType,
    limit: int = 10,
) -> ToolReturn:
    """Execute a search to find and rank entities.

    Use this tool for SELECT action to find entities matching your criteria.
    For counting or computing statistics, use run_aggregation instead.

    If the structured search finds nothing, this automatically retries with a
    broader semantic search (filters dropped) so the user still gets the closest
    matches; the result description states when matches are approximate.
    """
    state = ctx.deps.state
    response, final_query, fallback_used = await _execute_search_with_fallback(
        state, entity_type, limit, db.session
    )

    description = _describe_results(len(response.results), entity_type, fallback_used)

    state.memory.record_tool_step(
        ToolStep(
            step_type="run_search",
            description=description,
            context={
                "query_id": state.query_id,
                "query_snapshot": final_query.model_dump(),
                "fallback_used": fallback_used,
                "search_type": response.metadata.search_type,
            },
        )
    )

    logger.debug(
        "Search completed",
        total_count=len(response.results),
        query_id=str(state.query_id),
        fallback_used=fallback_used,
    )

    result_rows = [
        ResultRow(
            group_values={"entity_id": r.entity_id, "title": r.entity_title, "entity_type": r.entity_type.value},
            aggregations={"score": r.score},
        )
        for r in response.results
    ]
    full_response = QueryResultsResponse(
        results=result_rows,
        total_results=len(result_rows),
        metadata=response.metadata,
        visualization_type=VisualizationType(type="table"),
    )
    artifact = QueryArtifact(
        query_id=str(state.query_id),
        total_results=len(result_rows),
        visualization_type=VisualizationType(type="table"),
        description=description,
        search_type=response.metadata.search_type,
    )
    return ToolReturn(return_value=full_response, metadata=artifact)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS — all `TestDescribeResults` and `TestExecuteSearchWithFallback` cases pass.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest`
Expected: PASS — in particular `tests/test_memory.py` and `tests/test_adapters.py` still pass (their `"Searched 5 subscriptions"` strings are hand-built `ToolStep` fixtures, not assertions on `run_search` output, so the changed production description does not affect them).

- [ ] **Step 6: Commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/tools/search.py tests/test_search.py
git commit -m "Add semantic-search fallback when structured search returns nothing"
```

---

## Task 5: Final verification (lint, format, types, full suite)

**Files:** none (verification only)

- [ ] **Step 1: Type-check**

Run: `uv run mypy src/`
Expected: Success, no issues. If mypy flags the `or`-chained return in `_run_semantic_fallback`, confirm both branches return `tuple[...] | None` so the expression's type is `tuple[...] | None` (it is).

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Format check**

Run: `uv run black --check .`
Expected: All files would be left unchanged. (If not, run `uv run black .`, then re-stage and amend is NOT allowed — make a new commit.)

- [ ] **Step 4: Full test suite**

Run: `uv run pytest`
Expected: All tests pass.

- [ ] **Step 5: Commit any formatting fixes (only if Step 3 changed files)**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add -A
git commit -m "Apply black formatting"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** Task 2 → spec §1 (lenient prompt). Task 4 → spec §2 (semantic fallback ladder) and the memory `ToolStep` signaling in §3. Task 1 + Task 4 → spec §3 (`search_type` on artifact + exact/approximate descriptions). Task 3 → spec §4 (skill metadata). Task 5 → verification.
- **Why the fallback logic is in helpers, not in `run_search`:** building a real `RunContext[StateDeps[SearchState]]` in a unit test is heavy and brittle. The helpers (`_describe_results`, `_attempt_semantic_query`, `_run_semantic_fallback`, `_execute_search_with_fallback`) contain all branching and are tested directly. `run_search` is a thin wrapper (orchestrate → build response/artifact) whose passthrough of `response.metadata.search_type` into the artifact is covered by `tests/test_artifacts.py` (field) plus the helper tests (value), and whose event-shaping is already exercised by `tests/test_adapters.py`.
- **`search_type` default `""`:** required so the unchanged `QueryArtifact(...)` call in `tools/aggregation.py:110` keeps working. Only `run_search` sets it explicitly; aggregation artifacts keep `""`.
- **Trigger is exactly 0:** `_execute_search_with_fallback` returns early (`fallback_used=False`) whenever `response.results` is non-empty or `state.user_input` is empty, matching the spec's "exactly 0 results" and "empty input skips fallback" rules.
- **Never breaks the happy path:** if every fallback attempt errors (`ValueError`) or yields no rows, `_run_semantic_fallback` returns `None` and the original empty structured `response` is returned unchanged.
