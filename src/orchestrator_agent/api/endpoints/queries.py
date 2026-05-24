# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""Query results proxy.

A ``QueryArtifact`` emitted on the OpenAI streaming wire carries only a
``query_id`` reference; chat surfaces (LibreChat fork) fetch the full row
set here. We delegate straight to orchestrator-core's existing endpoint
function — same DB, same response shape.
"""

from uuid import UUID

import structlog
from fastapi.routing import APIRouter
from orchestrator.api.api_v1.endpoints.search import get_query_results
from orchestrator.search.query.results import QueryResultsResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/{query_id}/results", response_model=QueryResultsResponse)
async def query_results(query_id: UUID) -> QueryResultsResponse:
    """Fetch full results for a stored query (search / aggregation / count)."""
    return await get_query_results(query_id)
