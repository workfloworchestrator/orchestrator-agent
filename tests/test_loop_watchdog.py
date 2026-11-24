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

"""A blocked event loop is reported with the stack of the code that blocks it.

A loop that is stuck cannot log anything about itself, which is how a pod can go silent for minutes while it spins
on one request. The watchdog is a thread beside the loop: it sees that the loop stopped turning and logs where.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest  # noqa: E402
from structlog.testing import capture_logs  # noqa: E402

from orchestrator_agent.loop_watchdog import LoopWatchdog  # noqa: E402
from orchestrator_agent.settings import AgentSettings  # noqa: E402

BLOCKED, RECOVERED = "Event loop blocked", "Event loop turning again"


def _the_code_that_blocks_the_loop(seconds: float) -> None:
    time.sleep(seconds)


@pytest.mark.parametrize(
    ("stall", "blocks_for", "reported"),
    [
        pytest.param(5.0, 0.0, False, id="a-loop-that-turns-is-not-reported"),
        pytest.param(0.2, 0.7, True, id="a-blocked-loop-is-reported"),
    ],
)
async def test_a_blocked_loop_is_reported_with_the_stack_that_blocks_it(
    stall: float, blocks_for: float, reported: bool
):
    watchdog = LoopWatchdog(stall_seconds=stall)
    with capture_logs() as logs:
        watchdog.start()
        await asyncio.sleep(0.1)  # the loop turns
        _the_code_that_blocks_the_loop(blocks_for)
        await asyncio.sleep(0.3)  # and turns again
        watchdog.stop()

    blocked = [entry for entry in logs if entry["event"] == BLOCKED]
    assert bool(blocked) is reported
    assert all(entry["blocked_seconds"] >= stall for entry in blocked)
    assert all("_the_code_that_blocks_the_loop" in entry["stack"] for entry in blocked)
    assert [entry["event"] for entry in logs][-1:] == ([RECOVERED] if reported else [])


async def test_a_loop_that_stays_blocked_is_reported_again_each_stall_period():
    watchdog = LoopWatchdog(stall_seconds=0.2)
    with capture_logs() as logs:
        watchdog.start()
        await asyncio.sleep(0.05)
        _the_code_that_blocks_the_loop(0.9)
        await asyncio.sleep(0.2)
        watchdog.stop()

    assert len([entry for entry in logs if entry["event"] == BLOCKED]) >= 2


async def test_stopping_ends_the_watchdog_thread():
    before = threading.active_count()
    watchdog = LoopWatchdog(stall_seconds=0.2)
    watchdog.start()
    assert threading.active_count() == before + 1

    watchdog.stop()

    assert threading.active_count() == before


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        pytest.param({}, 5.0, id="on-by-default"),
        pytest.param({"EVENT_LOOP_STALL_SECONDS": 0}, 0.0, id="zero-turns-it-off"),
    ],
)
def test_the_stall_threshold_is_a_setting(setting: dict[str, float], expected: float):
    assert AgentSettings.model_validate(setting).EVENT_LOOP_STALL_SECONDS == expected
