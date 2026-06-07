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

from datetime import date, datetime
from typing import Any

import structlog
from orchestrator.core.db import db
from orchestrator.core.search.core.types import EntityType, FilterOp, QueryOperation, UIType
from orchestrator.core.search.filters import DateRangeFilter, DateValueFilter, FilterTree
from orchestrator.core.search.filters.definitions import TypeDefinition, generate_definitions
from orchestrator.core.search.query.builder import ComponentInfo, LeafInfo, build_paths_query, process_path_rows
from orchestrator.core.search.query.exceptions import PathNotFoundError, QueryValidationError
from orchestrator.core.search.query.queries import AggregateQuery, CountQuery, SelectQuery
from orchestrator.core.search.query.validation import validate_filter_tree
from pydantic import BaseModel, ConfigDict
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.state import SearchState

logger = structlog.get_logger(__name__)


class PathsResponse(BaseModel):
    """Response model for path discovery."""

    leaves: list[LeafInfo]
    components: list[ComponentInfo]

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


def ensure_query_initialized(
    state: SearchState,
    entity_type: EntityType,
    query_operation: QueryOperation = QueryOperation.SELECT,
) -> None:
    """Lazy-initialize query on state if not already set, or re-create if type mismatches.

    Preserves filters (from existing query or pending_filters) when creating/upgrading.
    """
    expected_types = {
        QueryOperation.SELECT: SelectQuery,
        QueryOperation.COUNT: CountQuery,
        QueryOperation.AGGREGATE: AggregateQuery,
    }
    expected_type = expected_types[query_operation]

    if state.query is not None and isinstance(state.query, expected_type):
        return

    if state.query is not None:
        logger.warning(
            "Query type mismatch — re-creating query, grouping/aggregations will be lost",
            existing_type=type(state.query).__name__,
            requested_operation=query_operation.value,
        )

    # Collect filters: from existing query, or from pending_filters set by set_filter_tree
    filters = (state.query.filters if state.query else None) or state.pending_filters

    if query_operation == QueryOperation.SELECT:
        state.query = SelectQuery(entity_type=entity_type, query_text=state.user_input, filters=filters)
    elif query_operation == QueryOperation.COUNT:
        state.query = CountQuery(entity_type=entity_type, filters=filters)
    else:
        state.query = AggregateQuery(entity_type=entity_type, aggregations=[], filters=filters)

    state.pending_filters = None


filter_building_toolset: FunctionToolset[StateDeps[SearchState]] = FunctionToolset(max_retries=2)


def _to_datetime(value: datetime | date | str) -> datetime | date:
    """Parse an ISO date/datetime string to a ``datetime``; pass non-strings through unchanged."""
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _coerce_condition_dates(condition: object) -> None:
    """Coerce a date filter leaf's string value(s) to ``datetime`` in place."""
    match condition:
        case DateValueFilter():
            condition.value = _to_datetime(condition.value)
        case DateRangeFilter():
            condition.value.start = _to_datetime(condition.value.start)
            condition.value.end = _to_datetime(condition.value.end)
        case _:
            pass


def _coerce_date_filter_values(filters: FilterTree) -> None:
    """Coerce ISO date strings in date filters to ``datetime`` so they bind as TIMESTAMP.

    Workaround for orchestrator-core (<= 5.0.4): ``DateValueFilter``/``DateRangeFilter`` compare a
    ``timestamptz``-cast column against the raw value. A ``str`` value binds as VARCHAR, so Postgres
    rejects ``timestamptz >= varchar``; a ``datetime`` binds as TIMESTAMP. Mutates the leaves in place.
    """
    for leaf in filters.get_all_leaves():
        _coerce_condition_dates(leaf.condition)


async def _list_paths(
    prefix: str,
    q: str | None,
    entity_type: EntityType,
    limit: int = 100,
) -> PathsResponse:
    """Query available filter paths from the database.

    Inlined from orchestrator.core.api.api_v1.endpoints.search.list_paths to avoid
    depending on the API layer.
    """
    stmt = build_paths_query(entity_type=entity_type, prefix=prefix, q=q)
    stmt = stmt.limit(limit)
    rows = db.session.execute(stmt).all()

    leaves, components = process_path_rows(rows)
    return PathsResponse(leaves=leaves, components=components)


async def _get_definitions() -> dict[UIType, TypeDefinition]:
    """Get type definitions with operator mappings.

    Inlined from orchestrator.core.api.api_v1.endpoints.search.get_definitions.
    """
    return generate_definitions()


@filter_building_toolset.tool
async def set_filter_tree(
    ctx: RunContext[StateDeps[SearchState]],
    filters: FilterTree | None,
    entity_type: EntityType,
) -> FilterTree | None:
    """Replace current filters atomically with a full FilterTree, or clear with None.

    See FilterTree model for structure, operators, and examples.
    Filters are validated immediately and applied when the query executes.
    """
    logger.debug(
        "Setting filter tree",
        entity_type=entity_type.value,
        has_filters=filters is not None,
        filter_summary=f"{len(filters.get_all_leaves())} filters" if filters else "no filters",
    )

    try:
        await validate_filter_tree(filters, entity_type)
    except PathNotFoundError as e:
        logger.debug(f"{PathNotFoundError.__name__}: {str(e)}")
        raise ModelRetry(f"{str(e)} Use discover_filter_paths tool to find valid paths.")
    except QueryValidationError as e:
        logger.debug(f"Query validation failed: {str(e)}")
        raise ModelRetry(str(e))
    except Exception as e:
        logger.error("Unexpected Filter validation exception", error=str(e))
        raise ModelRetry(f"Filter validation failed: {str(e)}. Please check your filter structure and try again.")

    if filters is not None:
        _coerce_date_filter_values(filters)

    # Store validated filters — applied when query is created by run_search/run_aggregation
    if ctx.deps.state.query is not None:
        ctx.deps.state.query = ctx.deps.state.query.model_copy(update={"filters": filters})
    else:
        ctx.deps.state.pending_filters = filters

    return filters


async def _discover_field(field_name: str, entity_type: EntityType) -> dict[str, Any]:
    """Discover the filterable leaves/components whose name contains ``field_name``."""
    paths_response = await _list_paths(prefix="", q=field_name, entity_type=entity_type, limit=100)
    needle = field_name.lower()
    matching_leaves = [
        {"name": leaf.name, "value_kind": leaf.ui_types, "paths": leaf.paths}
        for leaf in paths_response.leaves
        if needle in leaf.name.lower()
    ]
    matching_components = [
        {"name": comp.name, "value_kind": comp.ui_types}
        for comp in paths_response.components
        if needle in comp.name.lower()
    ]
    if not matching_leaves and not matching_components:
        return {
            "status": "NOT_FOUND",
            "guidance": f"No filterable paths found containing '{field_name}'. Do not create a filter for this.",
            "leaves": [],
            "components": [],
        }
    return {
        "status": "OK",
        "guidance": f"Found {len(matching_leaves)} field(s) and {len(matching_components)} component(s) for '{field_name}'.",
        "leaves": matching_leaves,
        "components": matching_components,
    }


@filter_building_toolset.tool
async def discover_filter_paths(
    ctx: RunContext[StateDeps[SearchState]],
    field_names: list[str],
    entity_type: EntityType | None = None,
) -> dict[str, dict[str, Any]]:
    """Discovers available filter paths for a list of field names.

    Returns a dictionary where each key is a field_name from the input list and
    the value is its discovery result.
    """
    if not entity_type:
        if ctx.deps.state.query:
            entity_type = ctx.deps.state.query.entity_type
        else:
            raise ModelRetry("Entity type not specified and no query in state. Pass entity_type.")

    return {field_name: await _discover_field(field_name, entity_type) for field_name in field_names}


@filter_building_toolset.tool_plain
async def get_valid_operators() -> dict[str, list[FilterOp]]:
    """Gets the mapping of field types to their valid filter operators."""
    definitions = await _get_definitions()

    operator_map = {}
    for ui_type, type_def in definitions.items():
        key = ui_type.value

        if hasattr(type_def, "operators"):
            operator_map[key] = type_def.operators
    return operator_map
