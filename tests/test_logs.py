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

"""A logged exception is printed without the frames' local variables.

structlog's default console renderer prints exceptions with rich, showing every frame's locals. A frame that holds
a large tool result or message history then takes minutes of CPU to render, on the event loop: in production a
single ``logger.exception`` stalled a pod until it was deleted.
"""

from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from structlog.dev import ConsoleRenderer, plain_traceback

from orchestrator_agent.logs import configure_logging


@pytest.fixture(autouse=True)
def _default_structlog() -> Iterator[None]:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


class _CountedRepr:
    """A local whose ``repr`` is counted: rendering locals would call it."""

    calls = 0

    def __repr__(self) -> str:
        type(self).calls += 1
        return "<huge>"


def _render_failure() -> str:
    renderer = structlog.get_config()["processors"][-1]
    big = _CountedRepr()  # noqa: F841 - a local of the failing frame, as a tool result would be
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        return str(
            renderer(None, "error", {"event": "Task failed", "exc_info": (type(error), error, error.__traceback__)})
        )


def test_an_exception_is_logged_without_rendering_locals() -> None:
    _CountedRepr.calls = 0
    configure_logging()

    output = _render_failure()

    assert "RuntimeError: boom" in output
    assert _CountedRepr.calls == 0


@pytest.mark.parametrize("times", [pytest.param(1, id="once"), pytest.param(2, id="twice")])
def test_only_the_exception_formatter_changes(times: int) -> None:
    defaults = structlog.get_config()["processors"]

    for _ in range(times):
        configure_logging()

    processors: list[Any] = structlog.get_config()["processors"]
    assert [type(p) for p in processors] == [type(p) for p in defaults]
    match processors[-1]:
        case ConsoleRenderer() as renderer:
            assert renderer._exception_formatter is plain_traceback
        case other:
            pytest.fail(f"the last processor is {other!r}, not a ConsoleRenderer")
