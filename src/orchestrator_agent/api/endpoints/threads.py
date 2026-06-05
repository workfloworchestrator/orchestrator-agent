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

"""Per-thread artifact collector.

Returns the artifact descriptors emitted during a turn, keyed by the A2A
``thread_id`` (context id). A surface (the proxy) reads this at end-of-turn and
injects the ``wfo-artifact:`` fences — the side channel that survives a
declarative orchestrator dropping the in-band DataParts.
"""

from typing import Any

from fastapi.routing import APIRouter

from orchestrator_agent.artifacts_store import list_artifacts

router = APIRouter()


@router.get("/{thread_id}/queries")
async def thread_queries(thread_id: str) -> list[dict[str, Any]]:
    """List ``{name, data}`` artifact descriptors emitted under ``thread_id``."""
    return list_artifacts(thread_id)
