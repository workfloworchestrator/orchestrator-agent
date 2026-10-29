# Identifier-aware search: configurable domain prompt + LLM-selectable retriever

**Date:** 2026-06-03
**Status:** Approved

## Problem

Users frequently reference entities by a concrete **identifier**: a customer name, a subscription id, or an opaque code/number such as `IS4443`, `4433`, or `id 1234`. These tokens are the highest-signal part of the request, but the search agent under-uses them:

- The agent's primary search sets `query_text = state.user_input` (the **full** user message) and lets orchestrator-core auto-route the retriever. Auto-routing (`Retriever._plan`, `retrieval/retrievers/base.py:59-72`) picks **SEMANTIC** for any multi-word `query_text`, **FUZZY** for a single token, **HYBRID** only for a single non-UUID word. So an identifier embedded in a sentence ("find the subscription with code IS4443") gets **SEMANTIC** ranking — the weakest mode for matching an exact code.
- **HYBRID** (Reciprocal-Rank-Fusion of semantic + fuzzy keyword) is the right mode for identifier-bearing queries, and on an *explicit* override it fuzzy-matches the full `query_text` (`base.py:163`). But `run_search` (`tools/search.py:149`) exposes **no retriever control** to the LLM, so HYBRID is unreachable for normal requests.
- There is no place for **deployment-specific domain knowledge** (e.g. "`IS####` is a circuit code → filter field `imsCircuitId`"). Such mappings vary per WFO install and must not be hardcoded.

## Goal

Make search identifier-aware via three coordinated changes:

1. A **configurable, env-driven domain-knowledge** section injected into the search prompt, so operators can teach the agent site-specific identifier→field conventions without code changes.
2. **Baseline (generic) prompt guidance** to extract identifiers from the request, discover the matching field via the existing autocomplete tool, and filter leniently — so good behavior ships out of the box.
3. An **LLM-selectable, embedding-aware retriever** on `run_search`, so the agent can choose **HYBRID** for identifier-centric queries and **SEMANTIC** for descriptive ones, degrading to **FUZZY** when embeddings are unavailable.

## Decisions (from brainstorming)

- **Placement:** identifier/retriever guidance lives in the **search-execution prompt** (`get_search_execution_prompt` / shared `FILTERING_RULES`), not the planner. The planner already routes these requests to SEARCH.
- **Ambiguous identifiers:** when the agent cannot confidently map an identifier to a field, it does **not** invent a filter — `query_text` already carries the token into ranking, and the existing no-result semantic fallback broadens. (It still *attempts* discovery first; "query-text only" is the fallback, not the first move.)
- **Prompt shape:** **baseline generic guidance + a configurable domain section** (not config-only, not generic-only).
- **Retriever selection is LLM-driven**, guided by the prompt and the operator's domain context — not heuristic code-level identifier detection (too brittle).
- **Embeddings may be disabled.** HYBRID = SEMANTIC + FUZZY, and SEMANTIC/HYBRID require embeddings (`llm_settings.EMBEDDING_API_ENABLED`). When embeddings are off, the agent must use FUZZY/structured and must never request an embedding retriever (which raises `ValueError` in `Retriever.route`, `base.py:148-151`).

## Relevant facts about the search machinery (verified against orchestrator-core 5.0.2)

- `RetrieverType` (`orchestrator.core.search.core.types`) has exactly: `FUZZY`, `SEMANTIC`, `HYBRID`.
- `SelectQuery` (`orchestrator.core.search.query.queries`) carries `query_text`, `retriever: RetrieverType | None`, `filters`. `SearchMixin` derives `vector_query` (the embeddable text; `None` if `query_text` is empty or a UUID) and `fuzzy_term` (the text only when it is a single word) — these drive **auto-routing** (`retriever is None`). A validator requires `query_text` to be set whenever `retriever` is not `None` (`query/mixins.py:80-83`).
- Embedding availability: `from orchestrator.core.settings import llm_settings` → `llm_settings.EMBEDDING_API_ENABLED` (a `bool`).
- The agent builds the primary query in `ensure_query_initialized` (`tools/filters.py:75`): `SelectQuery(entity_type=..., query_text=state.user_input, filters=filters)` — `retriever` is left `None` (auto).
- `_execute_search_with_fallback` (`tools/search.py:113-146`) runs the primary pass, then on zero rows runs `_run_semantic_fallback` (SEMANTIC, then auto). The fallback **already** tolerates embeddings-off (the SEMANTIC attempt raises `ValueError`, is caught, and degrades to the auto/fuzzy attempt) — **no change needed there**.
- `discover_filter_paths` (`tools/filters.py`) is the autocomplete/field-discovery tool the LLM already has; `FILTERING_RULES` already mandates calling it before `set_filter_tree` and already prefers `like` for text/partial values.

## Design

### Component 1 — Configurable domain-knowledge prompt

**Settings** (`src/orchestrator_agent/settings.py`, in `AgentSettings`):

```python
AGENT_DOMAIN_CONTEXT: str = Field(
    default="",
    description="Optional operator-supplied domain knowledge injected into the search prompt "
    "(e.g. identifier conventions and their filter fields). Empty disables the section.",
)
```

**Injection** (`get_search_execution_prompt`, `src/orchestrator_agent/prompts.py`): when `agent_settings.AGENT_DOMAIN_CONTEXT.strip()` is non-empty, render a dedicated section; when empty, render nothing (no empty header). A small pure helper keeps the f-string clean:

```python
def _domain_context_section() -> str:
    """Return a '## Domain Knowledge' block when AGENT_DOMAIN_CONTEXT is set, else ''."""
    context = agent_settings.AGENT_DOMAIN_CONTEXT.strip()
    if not context:
        return ""
    return f"## Domain Knowledge\n{context}\n"
```

The section is inserted into the search prompt body between the `## Steps` block and `FILTERING_RULES` (so domain conventions are visible before the agent builds filters). Example operator value:

```
Circuit codes look like "IS1234" and map to the imsCircuitId field — filter with like.
Customer references are 8-digit numbers — field customerId. Prefer HYBRID for these.
```

Scope: search prompt only (matches the placement decision). Extending the same injection to `get_aggregation_execution_prompt` is a trivial future follow-up and is **out of scope** here.

### Component 2 — Baseline identifier guidance (generic, shared)

Add an identifier bullet to `FILTERING_RULES` (`prompts.py:24`), which is shared by the search and aggregation prompts (both build filters):

```
- **EXTRACT IDENTIFIERS**: Scan the request for specific identifiers the user gave — entity/subscription ids, customer names, reference codes (e.g. `IS4443`), or numbers (e.g. `4433`, `id 1234`). These are the highest-signal part of the request. Use `{discover_filter_paths}` to find the field that holds such a value and filter it with `like` (substring/typo-tolerant). If no discovered field clearly matches an opaque identifier, do NOT invent a filter — the search already ranks on the full request text and will surface the closest matches. Never silently ignore an identifier the user provided.
```

(`{discover_filter_paths}` is the existing `{tools.discover_filter_paths.__name__}` interpolation already used elsewhere in `FILTERING_RULES`.)

### Component 3 — LLM-selectable, embedding-aware retriever

**Pure coercion helper** (`tools/search.py`), unit-testable without a DB:

```python
def _effective_retriever(requested: RetrieverType | None) -> RetrieverType | None:
    """Resolve the retriever to actually use, accounting for embedding availability.

    SEMANTIC and HYBRID need embeddings; when EMBEDDING_API_ENABLED is False they would
    raise ValueError in Retriever.route. Degrade those to FUZZY (which still keyword-matches
    the identifier). FUZZY and None (auto-routing) pass through unchanged.
    """
    if requested in (RetrieverType.SEMANTIC, RetrieverType.HYBRID) and not llm_settings.EMBEDDING_API_ENABLED:
        return RetrieverType.FUZZY
    return requested
```

**Thread the retriever through the primary pass.** `run_search` gains a parameter; `_execute_search_with_fallback` gains a matching parameter and applies it via the existing `model_copy`:

```python
@search_execution_toolset.tool
async def run_search(
    ctx: RunContext[StateDeps[SearchState]],
    entity_type: EntityType,
    limit: int = 10,
    retriever: RetrieverType | None = None,
) -> ToolReturn:
    """... (existing docstring) ...

    retriever: Ranking strategy. HYBRID (semantic + fuzzy keyword) for queries that
        center on an identifier/code/name; SEMANTIC for descriptive phrases; FUZZY for
        exact tokens or when embeddings are unavailable. Omit to let the system auto-route.
    """
    state = ctx.deps.state
    response, final_query, fallback_used = await _execute_search_with_fallback(
        state, entity_type, limit, db.session, retriever
    )
    ...  # unchanged below
```

In `_execute_search_with_fallback`, fold the coerced retriever into the existing `model_copy`. Computing `effective` as `None` when `state.user_input` is empty keeps the `validate_retriever_requires_query_text` validator satisfied (a `None` retriever is always valid, so it can be set unconditionally):

```python
async def _execute_search_with_fallback(
    state: SearchState,
    entity_type: EntityType,
    limit: int,
    session: WrappedSession,
    retriever: RetrieverType | None = None,
) -> tuple[SearchResponse, SelectQuery, bool]:
    ensure_query_initialized(state, entity_type)
    effective = _effective_retriever(retriever) if state.user_input else None
    query = cast(SelectQuery, cast(Query, state.query).model_copy(update={"limit": limit, "retriever": effective}))
    ...  # rest unchanged (execute, fallback on zero rows)
```

Default `retriever=None` reproduces today's behavior exactly (auto-routing), so the change is backward compatible. The semantic fallback is unchanged.

**Retriever guidance in the search prompt**, conditioned on embedding availability. Add a step/note to `get_search_execution_prompt` that reads `llm_settings.EMBEDDING_API_ENABLED`:

- Embeddings **enabled**: "When the request centers on an identifier/code/name, pass `retriever=HYBRID` to `run_search`. For descriptive or sentence-like requests use `retriever=SEMANTIC`. Omit `retriever` to auto-route."
- Embeddings **disabled**: "Embeddings are unavailable; rely on filters and `retriever=FUZZY` for identifier matching. Do not request SEMANTIC or HYBRID."

A small pure helper builds this string so it is unit-testable:

```python
def _retriever_guidance() -> str:
    """Retriever-selection guidance for the search prompt, conditioned on embeddings."""
    if llm_settings.EMBEDDING_API_ENABLED:
        return (
            "- **CHOOSE A RETRIEVER**: For identifier/code/name-centric requests pass "
            f"`retriever=HYBRID` to `{tools.run_search.__name__}`; for descriptive/sentence "
            "requests use `retriever=SEMANTIC`; omit it to auto-route."
        )
    return (
        "- **CHOOSE A RETRIEVER**: Embeddings are unavailable — use `retriever=FUZZY` for "
        f"identifier matching with `{tools.run_search.__name__}` and rely on filters. "
        "Do NOT request SEMANTIC or HYBRID."
    )
```

This helper's output is inserted into the search prompt's `## Steps` (near step 3, where `run_search` is called).

## Error handling & edge cases

- **Embeddings off + LLM requests SEMANTIC/HYBRID:** `_effective_retriever` coerces to FUZZY before the query is built, so `Retriever.route` never raises. The prompt also steers the LLM away from requesting them. Belt and suspenders.
- **Empty `state.user_input`:** the retriever is left `None` (auto) to satisfy `validate_retriever_requires_query_text`; mirrors the existing `not state.user_input` guard in the fallback.
- **Zero results after an explicit retriever:** the existing semantic fallback still runs (and itself tolerates embeddings-off). Unchanged.
- **Empty `AGENT_DOMAIN_CONTEXT`:** no Domain Knowledge section is emitted (no dangling header).
- **Validator/limit interaction:** `model_copy` keeps `limit` handling identical; only `retriever` is conditionally added.

## Tests

- **`tests/test_settings.py`** — `AGENT_DOMAIN_CONTEXT` defaults to `""`.
- **`tests/test_prompts.py`**:
  - `FILTERING_RULES`/search prompt contains the identifier-extraction guidance (assert key phrases: "EXTRACT IDENTIFIERS", `discover_filter_paths` name, "like").
  - Domain section: with `agent_settings.AGENT_DOMAIN_CONTEXT` monkeypatched to a sentinel, the search prompt contains `## Domain Knowledge` + the sentinel; with `""`, neither appears. (Two cases — use `@pytest.mark.parametrize`.)
  - Retriever guidance conditioned on embeddings: monkeypatch `llm_settings.EMBEDDING_API_ENABLED` True/False and assert HYBRID/SEMANTIC are mentioned vs. the FUZZY-only wording. (`@pytest.mark.parametrize` over the two states with `pytest.param(..., id=...)`.)
- **`tests/test_search.py`** — `_effective_retriever` (pure), monkeypatching `llm_settings.EMBEDDING_API_ENABLED`, `@pytest.mark.parametrize`'d:
  - embeddings on: `HYBRID→HYBRID`, `SEMANTIC→SEMANTIC`, `FUZZY→FUZZY`, `None→None`.
  - embeddings off: `HYBRID→FUZZY`, `SEMANTIC→FUZZY`, `FUZZY→FUZZY`, `None→None`.
  - And a `run_search`/`_execute_search_with_fallback` test (mirroring `tests/test_search.py:177-209`, faking `db` and monkeypatching `execute_search_with_persistence` to capture the query) asserting the coerced `retriever` is set on the executed `SelectQuery` for an explicit HYBRID request (embeddings on), and that `retriever=None` leaves it unset.

No LLM/integration test (matches repo convention; prompt changes are verified by text assertions and the suite staying green).

## Out of scope

- Injecting `AGENT_DOMAIN_CONTEXT` into the aggregation prompt (trivial follow-up).
- Heuristic code-level identifier detection / auto-forcing HYBRID without LLM choice.
- Letting the LLM set/refine `query_text` (it remains `state.user_input`).
- A `retriever` parameter on `run_aggregation` (aggregations use structured/grouped queries).
- Any change to the existing semantic fallback or to `get_entity_by_id`.
