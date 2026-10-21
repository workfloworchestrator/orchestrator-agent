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

from typing import Annotated

from ag_ui.core import RunAgentInput
from fastapi import APIRouter, Depends, HTTPException, Request
from orchestrator.core.db import db
from pydantic import ValidationError
from pydantic_ai.ui import SSE_CONTENT_TYPE
from starlette.responses import Response, StreamingResponse
from structlog import get_logger

from orchestrator_agent.adapters import AGUIWorker
from orchestrator_agent.adapters.ag_ui import ThreadForbiddenError
from orchestrator_agent.agent import WFOAgent
from orchestrator_agent.api.dependencies import get_agent
from orchestrator_agent.security import subject_of

router = APIRouter()
logger = get_logger(__name__)


@router.post("/")
async def agent_conversation(
    request: Request,
    agent: Annotated[WFOAgent, Depends(get_agent)],
) -> Response:
    """Agent conversation endpoint using pydantic-ai AG-UI protocol."""
    try:
        body = await request.json()
        run_input = RunAgentInput(**body)
    except ValidationError as e:
        logger.error("Invalid request body", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    auth_header = request.headers.get("authorization", "")
    auth_token = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else None

    try:
        stream = await AGUIWorker.run_request(
            agent=agent,
            run_input=run_input,
            db_session=db.session,
            auth_token=auth_token,
            owner=subject_of(request),
        )
    except ThreadForbiddenError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return StreamingResponse(stream, media_type=SSE_CONTENT_TYPE)
