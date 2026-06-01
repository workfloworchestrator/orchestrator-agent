# Lenient filtering and semantic-search fallback

**Date:** 2026-06-01
**Status:** Approved

## Problem

The search agent almost always builds a strict structured query and returns nothing when the exact values don't match. Two causes:

1. **Over-strict filters.** `FILTERING_RULES` in `src/orchestrator_agent/prompts.py:24` pushes the agent hard toward exact structured filters, and the LLM defaults to the `EQ` operator even for human-typed text (names, partial values) and for date/number ranges. An `EQ` on a free-text field that's slightly off yields zero candidates.

2. **No recovery when filters eliminate every candidate.** `run_search` (`src/orchestrator_agent/tools/search.py:38`) sets `query_text=user_input` and the LLM-built `filters`, with `retriever=None`. Filters restrict the candidate set first; the retriever only ranks within it. If the strict filters match nothing, the result is empty and the turn ends with no results.

We want searches to be more lenient by default and to fall back to a broad semantic search as a last resort, so the likelihood of returning *something* useful is high.

## Goal

Make searches far less likely to return nothing, reusing capabilities already present in `orchestrator-core` (no new retrieval engine), while never overriding a precise structured match when one exists.

## Key facts from orchestrator-core

- `SelectQuery` (`orchestrator/core/search/query/queries.py`) already supports `query_text`, a `retriever` override (`FUZZY` / `SEMANTIC` / `HYBRID`), and structured `filters`.
- Operators available per field type (`orchestrator/core/search/filters/definitions.py`):
  - **String**: `EQ`, `NEQ`, `LIKE`
  - **Number (int/float)**: `EQ`, `NEQ`, `LT`, `LTE`, `GT`, `GTE`, `BETWEEN`
  - **Datetime**: `EQ`, `NEQ`, `LT`, `LTE`, `GT`, `GTE`, `BETWEEN`
  - **Boolean**: `EQ`, `NEQ`
- `SemanticRetriever` (`retrieval/retrievers/semantic.py`) applies **no similarity threshold** — it ranks all embedded candidates by L2 distance and returns the closest N. With filters dropped and embeddings populated, it is effectively guaranteed non-empty.
- `FuzzyRetriever` (`retrieval/retrievers/fuzzy.py`) uses a trigram `<%` threshold, so it *can* return empty. Hence semantic — not fuzzy — is the primary last resort.
- `Retriever.route` (`retrieval/retrievers/base.py`): an explicit `retriever=SEMANTIC` raises `ValueError` if the query embedding is unavailable; auto-routing (`retriever=None`) instead degrades gracefully to fuzzy.
- Embeddings are configured in the target deployment (`EMBEDDING_API_ENABLED` on, `AiSearchIndex.embedding` populated), so vector semantic search is real, not a fuzzy stand-in.

## Design

Two coordinated changes: prompt guidance for lenient filter construction, and a deterministic code safety-net in `run_search`.

### 1. Lenient filter construction (prompt-driven) — `prompts.py`

Augment the shared `FILTERING_RULES` (used by both the search and aggregation prompts) with an operator-selection guide:

- Text, names, titles, descriptions, partial values → `LIKE` (e.g. `%acme%`), **not** `EQ`.
- Dates / numbers ("in 2025", "after X", "between X and Y", "more than 100") → range operators `BETWEEN` / `GT` / `GTE` / `LT` / `LTE`.
- Exact identifiers (UUIDs), enum/status values, booleans → `EQ`.
- Rule of thumb: prefer the most lenient operator that still captures the user's intent; avoid `EQ` on human-typed text.

Add one line to the **search** prompt only (`get_search_execution_prompt`, not the aggregation prompt): if a search finds nothing, the system automatically retries with a broader semantic search, so don't over-constrain — and when reporting, mention if a fallback was used.

The aggregation prompt gets the operator guidance (shared `FILTERING_RULES`) but **not** the fallback note: a count of 0 is a valid answer, not a miss.

No tool signatures change; the agent already calls `discover_filter_paths` → `get_valid_operators` → `set_filter_tree`, all of which already expose `LIKE` and the range operators.

### 2. Semantic fallback (code safety-net) — `tools/search.py`

In `run_search`, after the structured pass, when it returns **exactly 0** results and `state.user_input` is non-empty, run one fallback pass:

```python
SelectQuery(
    entity_type=entity_type,
    query_text=state.user_input,
    filters=None,
    retriever=RetrieverType.SEMANTIC,
    limit=limit,
)
```

executed through the existing `execute_search_with_persistence` path so the fallback query is persisted and gets its own `run_id` / `query_id` (the AG-UI frontend fetches full data by `query_id`).

Robustness ladder:

1. Structured pass (LLM filters). If results > 0 → return as today, no fallback.
2. If 0 and `user_input` present → semantic pass (`retriever=SEMANTIC`, `filters=None`).
3. If the semantic pass raises `ValueError` (embedding unavailable) → retry once with `retriever=None`, letting core auto-route and degrade to fuzzy.
4. If every fallback attempt errors or still yields 0 → return the original empty structured result. The fallback must never break the happy path.

The returned `query_id` and `QueryArtifact` point to whichever query produced the rows that are shown.

### 3. Signal exact vs. approximate matches

Every search result must make clear whether it is a direct filter match or an approximate (semantic/fuzzy) match, so users and the LLM don't mistake "closest related" rows for exact hits.

- **Source of truth:** `SearchResponse.metadata.search_type` already reports `structured` / `semantic` / `fuzzy` / `hybrid`. It flows into `QueryResultsResponse` returned to LLM/A2A/MCP consumers unchanged.
- **Surface it inline to the frontend:** add a `search_type: str` field to `QueryArtifact` (`src/orchestrator_agent/artifacts.py`), populated from `search_response.metadata.search_type`. Today the lightweight artifact streamed to AG-UI carries only `description` / `query_id` / `total_results`; the strategy is only visible after the REST fetch. Adding the field lets the UI badge results as exact vs. approximate without the round-trip.
- **Human-readable `description` distinguishes the two cases:**
  - Structured pass (filters matched): `"Found {n} matching {entity_type}"`.
  - Fallback pass (filters dropped, semantic/fuzzy): `"No exact matches — showing {n} closest {entity_type} by similarity"`.
- **Memory:** the `ToolStep` recorded by `run_search` captures whether the fallback ran, the retriever used, and that filters were dropped, so the LLM can explain it and later turns know the strict filter found nothing.

### 4. Advertise the behavior in the SEARCH skill (`skills.py`)

Update the `TaskAction.SEARCH` entry in `SKILLS` (`src/orchestrator_agent/skills.py`) so the advertised capability (used for A2A skill discovery and planning) reflects what now happens:

- `description`: extend to note lenient matching and the semantic fallback, e.g. `"Find subscriptions, products, workflows, processes. Uses lenient filters (partial text and ranges) and falls back to semantic similarity search when no exact matches are found."`
- `tags`: add `"fuzzy"` and `"semantic"` alongside the existing `"search"`, `"query"`.

This keeps the externally-visible skill metadata honest about the lenient + fallback behavior.

## Error handling & scope

- Trigger is **exactly 0** structured results — a non-empty structured result is always trusted and never overridden.
- Empty `state.user_input` → skip the fallback (nothing to embed or rank).
- Aggregation and count flows are unchanged.
- No new settings flag. A `SEARCH_SEMANTIC_FALLBACK` toggle on `AgentSettings` could disable the behavior per-deployment, but it is out of scope unless requested.

## Tests

`pytest` / `pytest-asyncio`, mocking `execute_search_with_persistence` in `tests/`:

1. Structured pass returns results → fallback **not** invoked; structured response returned; artifact `search_type` reflects the structured strategy and `description` reads as an exact-match phrasing.
2. Structured pass returns 0 (with `user_input` set) → fallback invoked with `filters=None`, `retriever=SEMANTIC`, `query_text=user_input`; fallback rows returned; response metadata and artifact `search_type` report `semantic`; `description` uses the "no exact matches — closest by similarity" phrasing.
3. Structured pass returns 0 and `user_input` empty → no fallback; empty result returned.
4. Semantic fallback raises `ValueError` → degrades to auto-route (`retriever=None`); if all attempts empty, the original empty structured result is returned (happy path preserved).
5. Light assertion that `FILTERING_RULES` contains the operator-selection guidance (e.g. mentions `LIKE` and a range operator).
6. The `SEARCH` skill `description` mentions the fallback and its `tags` include `"semantic"` (guards against the advertised metadata drifting from behavior).

## Out of scope

- Building any new retriever or changing core ranking.
- Exposing `retriever` / `query_text` as LLM-controlled `run_search` parameters (the fallback is internal and automatic).
- Runtime rewriting of the LLM's filter tree (e.g. programmatic `EQ`→`LIKE`); leniency is achieved at construction time via the prompt, and the code path drops filters entirely on fallback.
- A "fewer than N results" trigger; fallback fires only on exactly 0.
- Per-deployment enable/disable toggle.
