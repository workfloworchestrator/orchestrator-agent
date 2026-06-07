# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import cast
from uuid import UUID

import structlog
from orchestrator.core.db import db
from orchestrator.core.db.database import WrappedSession
from orchestrator.core.search.core.types import EntityType, RetrieverType
from orchestrator.core.search.filters import FilterTree, PathFilter, StringFilter
from orchestrator.core.search.query.queries import Query, SelectQuery
from orchestrator.core.search.query.results import (
    QueryResultsResponse,
    ResultRow,
    SearchResponse,
    VisualizationType,
)
from orchestrator.core.settings import llm_settings
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.artifacts import QueryArtifact
from orchestrator_agent.handlers import execute_search_with_persistence
from orchestrator_agent.memory import ToolStep
from orchestrator_agent.settings import SearchEffort, agent_settings
from orchestrator_agent.state import SearchState
from orchestrator_agent.tools.filters import ensure_query_initialized

logger = structlog.get_logger(__name__)

search_execution_toolset: FunctionToolset[StateDeps[SearchState]] = FunctionToolset(max_retries=2)


class MatchMode(str, Enum):
    """How closely the returned rows match the user's original filters.

    EXACT — the full filter set matched (or every broadening pass was empty).
    RELAXED — the loose text filters were dropped but the high-signal exact filters
    (ids, status, customer) were kept, so the rows are still scoped, just broader.
    SIMILARITY — all filters were dropped and rows are the closest matches by ranking.
    """

    EXACT = "exact"
    RELAXED = "relaxed"
    SIMILARITY = "similarity"


# How many broadening rungs each effort level may try when a filtered search is empty.
# HIGH=3 so it can exhaust the full ladder (relaxed filters, then HYBRID, then SEMANTIC).
_EFFORT_FALLBACK_PASSES: dict[SearchEffort, int] = {
    SearchEffort.LOW: 0,
    SearchEffort.MEDIUM: 1,
    SearchEffort.HIGH: 3,
}


def _describe_results(count: int, entity_type: EntityType, mode: MatchMode) -> str:
    """Summarize results, telling the user how the rows relate to their filters."""
    label = entity_type.value
    match mode:
        case MatchMode.RELAXED:
            return f"No exact match on all criteria — showing {count} {label} matching the key filters"
        case MatchMode.SIMILARITY:
            return f"No exact matches — showing {count} closest {label} by similarity"
        case _:
            return f"Found {count} matching {label}"


def _is_relaxable(leaf: PathFilter) -> bool:
    """A loose `like`/text leaf — the first thing to drop when a filtered search is empty.

    Exact (`eq`/`neq`), range, and component filters carry high signal (ids, status,
    customer, dates) and are kept; only substring text matches are relaxed.
    """
    match leaf.condition:
        case StringFilter():
            return True
        case _:
            return False


def _high_signal_filters(filters: FilterTree | None) -> FilterTree | None:
    """Drop loose `like`/text leaves, keeping the high-signal exact/range leaves.

    Returns None when relaxing is not a useful step: no filters, nothing relaxable
    (the reduced tree would equal the original), or nothing high-signal to keep
    (the reduced tree would be filterless, i.e. the next rung). The kept leaves are
    rebuilt as a flat AND; any nested OR structure is intentionally flattened, which
    only ever broadens the candidate set.
    """
    if filters is None:
        return None
    leaves = filters.get_all_leaves()
    relaxable = [leaf for leaf in leaves if _is_relaxable(leaf)]
    high_signal = [leaf for leaf in leaves if not _is_relaxable(leaf)]
    if not relaxable or not high_signal:
        return None
    return FilterTree.from_flat_and(high_signal)


def _effective_retriever(requested: RetrieverType | None) -> RetrieverType | None:
    """Resolve the retriever to actually use, accounting for embedding availability.

    SEMANTIC and HYBRID need embeddings; when EMBEDDING_API_ENABLED is False they would
    raise ValueError in Retriever.route. Degrade those to FUZZY (which still keyword-matches
    the identifier). FUZZY and None (auto-routing) pass through unchanged.
    """
    if requested in (RetrieverType.SEMANTIC, RetrieverType.HYBRID) and not llm_settings.EMBEDDING_API_ENABLED:
        return RetrieverType.FUZZY
    return requested


async def _attempt_query(
    filters: FilterTree | None,
    retriever: RetrieverType | None,
    *,
    entity_type: EntityType,
    query_text: str,
    limit: int,
    run_id: UUID | None,
    session: WrappedSession,
) -> tuple[SearchResponse, UUID, UUID, SelectQuery] | None:
    """Run one broadening pass with the given filters/retriever; return it only if it produced rows.

    ``retriever`` is resolved through ``_effective_retriever`` so embedding-based strategies
    degrade to fuzzy when embeddings are unavailable. A ValueError means the embedding could
    not be generated for an explicit override — treat it as "no help" so the caller advances
    to the next rung.
    """
    query = SelectQuery(
        entity_type=entity_type,
        query_text=query_text,
        filters=filters,
        retriever=_effective_retriever(retriever),
        limit=limit,
    )
    try:
        response, new_run_id, query_id = await execute_search_with_persistence(query, session, run_id)
    except ValueError as exc:
        logger.debug("Broadening attempt unavailable", retriever=retriever, error=str(exc))
        return None
    return (response, new_run_id, query_id, query) if response.results else None


# An ordered broadening ladder consumed (up to the effort budget) when a filtered search is empty.
_BroadeningStep = tuple[FilterTree | None, RetrieverType | None, MatchMode]


def _broadening_ladder(filters: FilterTree | None) -> list[_BroadeningStep]:
    """Build the ordered broadening rungs for an empty filtered search.

    1. RELAXED — keep the high-signal exact filters, drop loose text filters (only if useful).
    2. SIMILARITY/HYBRID — drop all filters, keyword-match identifiers before pure semantics.
    3. SIMILARITY/SEMANTIC — drop all filters, pure semantic ranking as the last resort.
    """
    reduced = _high_signal_filters(filters)
    relaxed_rung: list[_BroadeningStep] = [(reduced, None, MatchMode.RELAXED)] if reduced is not None else []
    return relaxed_rung + [
        (None, RetrieverType.HYBRID, MatchMode.SIMILARITY),
        (None, RetrieverType.SEMANTIC, MatchMode.SIMILARITY),
    ]


async def _run_broadening_fallback(
    *,
    filters: FilterTree | None,
    entity_type: EntityType,
    query_text: str,
    limit: int,
    run_id: UUID | None,
    session: WrappedSession,
    passes: int,
) -> tuple[SearchResponse, UUID, UUID, SelectQuery, MatchMode] | None:
    """Try up to ``passes`` broadening rungs, returning the first that produced rows (with its mode).

    ``passes == 0`` disables broadening entirely.
    """
    for step_filters, step_retriever, step_mode in _broadening_ladder(filters)[:passes]:
        result = await _attempt_query(
            step_filters,
            step_retriever,
            entity_type=entity_type,
            query_text=query_text,
            limit=limit,
            run_id=run_id,
            session=session,
        )
        if result is not None:
            response, new_run_id, query_id, query = result
            return response, new_run_id, query_id, query, step_mode
    return None


async def _execute_search_with_fallback(
    state: SearchState,
    entity_type: EntityType,
    limit: int,
    session: WrappedSession,
    retriever: RetrieverType | None = None,
    *,
    effort: SearchEffort | None = None,
) -> tuple[SearchResponse, SelectQuery, MatchMode]:
    """Run the structured pass, then broaden progressively when it returns zero rows.

    The number of broadening passes is governed by ``effort`` (defaulting to the configured
    ``AGENT_SEARCH_EFFORT``): HIGH=3, MEDIUM=1, LOW=0. Broadening first relaxes the loose text
    filters while keeping the high-signal exact filters (RELAXED), then drops all filters and
    ranks by HYBRID, then SEMANTIC (SIMILARITY). Updates state.run_id/query_id to the query that
    produced the returned rows. Returns (response, final_query, match_mode).
    """
    resolved_effort = effort if effort is not None else agent_settings.AGENT_SEARCH_EFFORT
    ensure_query_initialized(state, entity_type)
    effective = _effective_retriever(retriever) if state.user_input else None
    query = cast(SelectQuery, cast(Query, state.query).model_copy(update={"limit": limit, "retriever": effective}))
    response, run_id, query_id = await execute_search_with_persistence(query, session, state.run_id)
    state.run_id = run_id
    state.query_id = query_id

    if response.results or not state.user_input:
        return response, query, MatchMode.EXACT

    fallback = await _run_broadening_fallback(
        filters=query.filters,
        entity_type=entity_type,
        query_text=state.user_input,
        limit=limit,
        run_id=state.run_id,
        session=session,
        passes=_EFFORT_FALLBACK_PASSES[resolved_effort],
    )
    if fallback is None:
        return response, query, MatchMode.EXACT

    fb_response, fb_run_id, fb_query_id, fb_query, fb_mode = fallback
    state.run_id = fb_run_id
    state.query_id = fb_query_id
    return fb_response, fb_query, fb_mode


@search_execution_toolset.tool
async def run_search(
    ctx: RunContext[StateDeps[SearchState]],
    entity_type: EntityType,
    limit: int | None = None,
    retriever: RetrieverType | None = None,
) -> ToolReturn:
    """Execute a search to find and rank entities.

    Use this tool for SELECT action to find entities matching your criteria.
    For counting or computing statistics, use run_aggregation instead.

    If the structured search finds nothing, this automatically broadens: first
    relaxing the loose text filters while keeping the high-signal exact filters
    (ids, status, customer), then dropping all filters and ranking by similarity.
    The result description states when matches are relaxed or approximate.

    Args:
        ctx: Tool run context providing access to agent state.
        entity_type: Type of entity to search (SUBSCRIPTION, PRODUCT, WORKFLOW, or PROCESS).
        limit: Maximum number of results to return. Omit to use the server-configured default;
            set a higher value when the user asks for more (e.g. "top 50", "first 100", or "all" —
            use a large number).
        retriever: Ranking strategy. HYBRID (semantic + fuzzy keyword) for queries centered on
            an identifier/code/name; SEMANTIC for descriptive phrases; FUZZY for exact tokens or
            when embeddings are unavailable. Omit to auto-route.
    """
    state = ctx.deps.state
    effective_limit = limit if limit is not None else agent_settings.SEARCH_RESULT_LIMIT
    response, final_query, match_mode = await _execute_search_with_fallback(
        state, entity_type, effective_limit, db.session, retriever
    )

    description = _describe_results(len(response.results), entity_type, match_mode)

    state.memory.record_tool_step(
        ToolStep(
            step_type="run_search",
            description=description,
            context={
                "query_id": state.query_id,
                "query_snapshot": final_query.model_dump(),
                "match_mode": match_mode.value,
                "search_type": response.metadata.search_type,
            },
        )
    )

    logger.debug(
        "Search completed",
        total_count=len(response.results),
        query_id=str(state.query_id),
        match_mode=match_mode.value,
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
