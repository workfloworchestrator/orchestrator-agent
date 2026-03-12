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

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from orchestrator.db import SearchQueryTable, db
from orchestrator.search.core.exceptions import QueryStateNotFoundError
from orchestrator.search.query import engine
from orchestrator.search.query.queries import AggregateQuery, CountQuery, ExportQuery, QueryAdapter, SelectQuery
from orchestrator.search.query.results import QueryResultsResponse, ResultRow, VisualizationType
from orchestrator.search.query.state import QueryState

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get(
    "/queries/{query_id}/results",
    response_model=QueryResultsResponse,
    summary="Fetch full query results by query_id",
)
async def get_query_results(query_id: UUID) -> QueryResultsResponse:
    """Fetch full results for any query type (select, count, aggregate).

    Detects query type from stored parameters and executes accordingly,
    always returning QueryResultsResponse for consistent client rendering.
    """
    try:
        row = db.session.query(SearchQueryTable).filter_by(query_id=query_id).first()
        if not row:
            raise QueryStateNotFoundError(f"Query {query_id} not found")

        query = QueryAdapter.validate_python(row.parameters)

        if isinstance(query, SelectQuery):
            embedding = list(row.query_embedding) if row.query_embedding is not None else None
            search_response = await engine.execute_search(query, db.session, query_embedding=embedding)
            result_rows = [
                ResultRow(
                    group_values={
                        "entity_id": result.entity_id,
                        "title": result.entity_title,
                        "entity_type": result.entity_type.value,
                    },
                    aggregations={"score": result.score},
                )
                for result in search_response.results
            ]
            return QueryResultsResponse(
                results=result_rows,
                total_results=len(result_rows),
                metadata=search_response.metadata,
                visualization_type=VisualizationType(type="table"),
            )

        if isinstance(query, (CountQuery, AggregateQuery)):
            return await engine.execute_aggregation(query, db.session)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported query type: {query.query_type}",
        )
    except QueryStateNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch query results", query_id=query_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch query results: {str(e)}",
        )


@router.get(
    "/queries/{query_id}/export",
    summary="Export query results by query_id",
)
async def export_by_query_id(query_id: str) -> dict:
    """Export search results using query_id.

    The query is retrieved from the database, re-executed, and results are returned
    as flattened records suitable for CSV download.
    """
    try:
        query_state = QueryState.load_from_id(query_id, SelectQuery)

        export_query = ExportQuery(
            entity_type=query_state.query.entity_type,
            filters=query_state.query.filters,
            query_text=query_state.query.query_text,
        )

        export_records = await engine.execute_export(export_query, db.session, query_state.query_embedding)
        return {"page": export_records}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except QueryStateNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Export failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing export: {str(e)}",
        )
