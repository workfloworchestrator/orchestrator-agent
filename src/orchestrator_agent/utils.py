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

from __future__ import annotations

from time import time_ns
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from orchestrator_agent.state import ExecutionPlan

logger = structlog.get_logger(__name__)


def current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return time_ns() // 1_000_000


def log_execution_plan(plan: "ExecutionPlan | None") -> None:
    if not plan:
        logger.debug("execution_plan", plan=None)
        return

    logger.debug(
        "execution_plan",
        num_tasks=len(plan.tasks),
        tasks=[
            {
                "index": i + 1,
                "status": task.status.value,
                "action": task.action_type.value,
                "reasoning": task.reasoning,
            }
            for i, task in enumerate(plan.tasks)
        ],
    )


def log_agent_request(node_name: str, instructions: str, message_history: list) -> None:
    logger.debug(
        "llm_request",
        node=node_name,
        instructions=instructions,
        message_history=[
            {
                "index": i,
                "kind": msg.kind,
                "parts": [
                    {
                        "type": part.__class__.__name__,
                        "content": part.content if hasattr(part, "content") else str(part),
                    }
                    for part in msg.parts
                ],
            }
            for i, msg in enumerate(message_history, 1)
        ],
    )
