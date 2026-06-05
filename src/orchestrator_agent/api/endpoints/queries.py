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

"""Query results proxy.

A ``QueryArtifact`` emitted on the wire carries only a ``query_id`` reference;
chat surfaces (the LibreChat fork's QueryArtifactCard) fetch the full row set
here. We delegate straight to orchestrator-core's existing endpoint function —
same DB, same response shape — so this is a thin path-bridge from the card's
``/queries/{id}/results`` to core's ``get_query_results``.

In production the agentgateway performs this bridge to orchestrator-core
directly; this endpoint is the local-dev equivalent so the card can render
without a gateway.
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
