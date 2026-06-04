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

# Ordered broadening passes tried (in order) when a filtered search returns nothing.
# SEMANTIC first because it applies no similarity threshold and is effectively guaranteed
# non-empty; None (auto-routing) is the wider net that degrades to fuzzy in orchestrator-core.
_FALLBACK_RETRIEVER_ORDER: tuple[RetrieverType | None, ...] = (RetrieverType.SEMANTIC, None)

# How many of the above passes each effort level is allowed to run.
_EFFORT_FALLBACK_PASSES: dict[SearchEffort, int] = {
    SearchEffort.LOW: 0,
    SearchEffort.MEDIUM: 1,
    SearchEffort.HIGH: 2,
}


def _describe_results(count: int, entity_type: EntityType, fallback_used: bool) -> str:
    """Summarize results, distinguishing exact filter matches from approximate ones."""
    label = entity_type.value
    if fallback_used:
        return f"No exact matches — showing {count} closest {label} by similarity"
    return f"Found {count} matching {label}"


def _effective_retriever(requested: RetrieverType | None) -> RetrieverType | None:
    """Resolve the retriever to actually use, accounting for embedding availability.

    SEMANTIC and HYBRID need embeddings; when EMBEDDING_API_ENABLED is False they would
    raise ValueError in Retriever.route. Degrade those to FUZZY (which still keyword-matches
    the identifier). FUZZY and None (auto-routing) pass through unchanged.
    """
    if requested in (RetrieverType.SEMANTIC, RetrieverType.HYBRID) and not llm_settings.EMBEDDING_API_ENABLED:
        return RetrieverType.FUZZY
    return requested


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
    except ValueError as exc:
        logger.debug("Semantic fallback attempt unavailable", retriever=retriever, error=str(exc))
        return None
    return (response, new_run_id, query_id, query) if response.results else None


async def _run_semantic_fallback(
    *,
    entity_type: EntityType,
    query_text: str,
    limit: int,
    run_id: UUID | None,
    session: WrappedSession,
    passes: int,
) -> tuple[SearchResponse, UUID, UUID, SelectQuery] | None:
    """Run up to ``passes`` filterless broadening searches, returning the first with rows.

    Tries the retrievers in ``_FALLBACK_RETRIEVER_ORDER`` (SEMANTIC, then auto-routed) up to
    ``passes`` times. SemanticRetriever applies no similarity threshold, so with filters dropped
    it returns the closest N embedded entities and is effectively guaranteed non-empty. If the
    explicit semantic pass cannot embed the query (ValueError) it yields nothing, so the
    auto-routed pass (retriever=None) — when allowed — degrades to fuzzy in orchestrator-core.
    ``passes == 0`` disables broadening entirely.
    """
    for retriever in _FALLBACK_RETRIEVER_ORDER[:passes]:
        result = await _attempt_semantic_query(
            retriever,
            entity_type=entity_type,
            query_text=query_text,
            limit=limit,
            run_id=run_id,
            session=session,
        )
        if result is not None:
            return result
    return None


async def _execute_search_with_fallback(
    state: SearchState,
    entity_type: EntityType,
    limit: int,
    session: WrappedSession,
    retriever: RetrieverType | None = None,
    *,
    effort: SearchEffort | None = None,
) -> tuple[SearchResponse, SelectQuery, bool]:
    """Run the structured pass, then a semantic fallback when it returns zero rows.

    The number of broadening fallback passes is governed by ``effort`` (defaulting to the
    configured ``AGENT_SEARCH_EFFORT``): HIGH=2, MEDIUM=1, LOW=0. Updates state.run_id/query_id
    to the query that produced the returned rows. Returns (response, final_query, fallback_used).
    """
    resolved_effort = effort if effort is not None else agent_settings.AGENT_SEARCH_EFFORT
    ensure_query_initialized(state, entity_type)
    effective = _effective_retriever(retriever) if state.user_input else None
    query = cast(SelectQuery, cast(Query, state.query).model_copy(update={"limit": limit, "retriever": effective}))
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
        passes=_EFFORT_FALLBACK_PASSES[resolved_effort],
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
    limit: int | None = None,
    retriever: RetrieverType | None = None,
) -> ToolReturn:
    """Execute a search to find and rank entities.

    Use this tool for SELECT action to find entities matching your criteria.
    For counting or computing statistics, use run_aggregation instead.

    If the structured search finds nothing, this automatically retries with a
    broader semantic search (filters dropped) so the user still gets the closest
    matches; the result description states when matches are approximate.

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
    response, final_query, fallback_used = await _execute_search_with_fallback(
        state, entity_type, effective_limit, db.session, retriever
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
