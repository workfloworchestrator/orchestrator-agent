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

import structlog
from fastapi import APIRouter
from fastapi.responses import Response
from orchestrator.db import db
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/")
def get_health() -> Response:
    """Health check — verifies database connectivity."""
    try:
        db.session.execute(text("SELECT 1"))
    except OperationalError as e:
        logger.warning("Health check failed", error=str(e))
        return Response(status_code=HTTP_503_SERVICE_UNAVAILABLE)
    return Response(status_code=HTTP_200_OK)
