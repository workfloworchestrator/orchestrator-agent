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

"""Run heavy synchronous work off the event loop.

Every request of the agent runs on one event loop. Processing a large tool result (parsing and rendering an
aggregation of tens of thousands of buckets, the history dump and load, the A2A artifact part) is CPU-bound,
synchronous Python: on the loop it holds up every other request, and the health endpoint with them.

``off_loop`` runs such a function on a worker thread of a pool of its own. A thread does not make the work faster,
because of the GIL: two CPU-bound threads share one core. What it changes is who waits. CPython hands the GIL to
another thread every few milliseconds (``sys.getswitchinterval``), so the loop thread gets its turn between the
chunks of the work and keeps serving the other requests and probes, where on the loop itself it would not run at
all until the work was done. A single call into a C extension that does not release the GIL (one JSON encode of a
very large document) is not interrupted either way, so it stays a block for the length of that call, no longer
than it takes to encode the one document. For work that must run in parallel, use more processes
(``WEB_CONCURRENCY``, see the runtime reference).

The pool is a small one of our own, so a burst of heavy work cannot take the threads the framework runs its
synchronous endpoints on (the ``/health`` route), nor the loop's default executor.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# More threads would only contend for the GIL: the work is CPU-bound Python, and a few make progress while
# the loop thread gets its turns.
OFFLOAD_THREADS = 4

_pool = ThreadPoolExecutor(max_workers=OFFLOAD_THREADS, thread_name_prefix="offload")


async def off_loop(func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    """``func(*args, **kwargs)`` run on a worker thread; its result, or what it raises.

    The function runs in a copy of the caller's context, so what a request binds (the person's token, the log
    context) is visible in it. It must not touch the event loop, and must not share mutable state with the loop
    thread beyond its arguments.
    """
    context = contextvars.copy_context()
    return await asyncio.get_running_loop().run_in_executor(_pool, partial(context.run, partial(func, *args, **kwargs)))


__all__ = ["OFFLOAD_THREADS", "off_loop"]
