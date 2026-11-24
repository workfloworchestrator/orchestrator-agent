# Copyright 2019-2026 SURF, GÉANT.
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

"""A large aggregation is processed in time proportional to its size, by each step that handles it.

Reproduction of the processing of a result of tens of thousands of buckets, through the real functions: the
artifact builder (table and chart), the history dump and load, and the A2A artifact part. Measured on a laptop each
takes well under a second; the budget is generous so that only a super-linear step (which would take minutes) fails.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest  # noqa: E402
from a2a.helpers import new_data_part  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from orchestrator_agent.capabilities.behavior.artifacts import query_artifact  # noqa: E402
from orchestrator_agent.persistence import dump_messages, load_messages  # noqa: E402

BUCKETS = 30_000
BUDGET_SECONDS = 10.0


def _aggregation(buckets: int) -> dict[str, Any]:
    """The aggregate tool's result, grouped by customer and port name: a bucket per row."""
    return {
        "query_id": "00000000-0000-0000-0000-000000000001",
        "total_results": buckets,
        "visualization": "bar",
        "results": [
            {
                "group_values": {"subscription.customer_id": f"customer-{i}", "port": f"Port number {i}"},
                "aggregations": {"count": i % 7 + 1, "sum_speed": float(i)},
            }
            for i in range(buckets)
        ],
    }


def _history_of(result: dict[str, Any]) -> list[Any]:
    """One earlier run: the person asks, the model calls ``aggregate``, the tool answers with ``result``."""
    return [
        ModelRequest(parts=[UserPromptPart(content="ports per customer")]),
        ModelResponse(parts=[ToolCallPart(tool_name="aggregate", args={}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="aggregate", content=result, tool_call_id="c1")]),
        ModelResponse(parts=[TextPart(content="Done.")]),
    ]


@pytest.mark.parametrize(
    "process",
    [
        pytest.param(lambda result: query_artifact("aggregate", {}, result), id="table"),
        pytest.param(lambda result: query_artifact("aggregate", {"visualization_type": "bar"}, result), id="chart"),
        pytest.param(lambda result: new_data_part(json.loads(json.dumps(result))), id="a2a-artifact-part"),
        pytest.param(lambda result: dump_messages(_history_of(result)), id="history-dump"),
        pytest.param(lambda result: load_messages(dump_messages(_history_of(result))), id="history-dump-and-load"),
    ],
)
def test_a_large_aggregation_is_processed_within_the_budget(process: Callable[[dict[str, Any]], Any]):
    result = _aggregation(BUCKETS)

    started = time.perf_counter()
    process(result)

    assert time.perf_counter() - started < BUDGET_SECONDS
