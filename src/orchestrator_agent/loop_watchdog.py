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

"""Report an event loop that is blocked, with the stack of the code that blocks it.

A blocked loop cannot say so itself: no request is served, no log line is written, and the readiness probe
times out, which is all anyone sees from outside. A thread beside the loop can. The loop stamps a heartbeat
every fraction of ``stall_seconds``; the watchdog thread notices when the stamp goes stale and logs the stack
the loop thread is in at that moment, again for every further ``stall_seconds`` the block lasts, so a long
block yields a sample of where the time goes. When the loop turns again it says so.

The watchdog is a thread, so it needs the GIL to look. It gets it between the bytecodes of pure Python (the
interpreter hands it over every few milliseconds), but not during a single call into a C extension that holds
it (a JSON encode of a huge document): that is reported when the call returns, with the stack of what runs next.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback

import structlog

logger = structlog.get_logger(__name__)

STACK_FRAMES = 30  # the innermost frames of the loop thread's stack that are logged


class LoopWatchdog:
    """Logs a warning with the loop thread's stack when the event loop has not turned for ``stall_seconds``."""

    def __init__(self, stall_seconds: float) -> None:
        self.stall_seconds = stall_seconds
        self._interval = min(0.5, stall_seconds / 4)  # how often the loop stamps, and the watchdog looks
        self._stamped = time.monotonic()
        self._loop_thread: int | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="loop-watchdog", daemon=True)

    def start(self) -> None:
        """Watch the running event loop (call from it, in its thread)."""
        self._loop_thread = threading.get_ident()
        self._stamp(asyncio.get_running_loop())
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._timer is not None:
            self._timer.cancel()
        if self._thread.is_alive():
            self._thread.join()

    def _stamp(self, loop: asyncio.AbstractEventLoop) -> None:
        self._stamped = time.monotonic()
        self._timer = loop.call_later(self._interval, self._stamp, loop)

    def _stack(self) -> str:
        frame = sys._current_frames().get(self._loop_thread or 0)
        return "".join(traceback.format_stack(frame, limit=STACK_FRAMES)) if frame is not None else ""

    def _watch(self) -> None:
        reported = 0  # the stall periods of the current block that are logged
        longest = 0.0
        while not self._stopped.wait(self._interval):
            blocked = time.monotonic() - self._stamped
            periods = int(blocked // self.stall_seconds)
            if periods > reported:
                logger.warning("Event loop blocked", blocked_seconds=round(blocked, 2), stack=self._stack())
                reported, longest = periods, blocked
            elif reported and periods == 0:
                logger.info("Event loop turning again", blocked_seconds=round(longest, 2))
                reported, longest = 0, 0.0


__all__ = ["LoopWatchdog"]
