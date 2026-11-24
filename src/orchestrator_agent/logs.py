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

"""Logging setup: structlog's defaults, with exceptions printed as plain tracebacks.

With rich installed, structlog's default console renderer prints an exception with every frame's local variables
(up to 100 frames). The agent's frames hold tool results and message histories, and rendering those took minutes
of CPU on the event loop in production: one ``logger.exception`` stalled the whole process. A plain traceback keeps
the stack and the error, and never looks at the locals.
"""

import structlog
from structlog.dev import ConsoleRenderer, plain_traceback
from structlog.typing import Processor


def _plain(processor: Processor) -> Processor:
    match processor:
        case ConsoleRenderer():
            return ConsoleRenderer(exception_formatter=plain_traceback)
        case _:
            return processor


def configure_logging() -> None:
    """Keep structlog's processors, with the console renderer printing exceptions without locals."""
    structlog.configure(processors=[_plain(processor) for processor in structlog.get_config()["processors"]])


__all__ = ["configure_logging"]
