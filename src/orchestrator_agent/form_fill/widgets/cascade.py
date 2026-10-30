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

"""A widget field asked in steps: first what narrows it down, then its value — a node, then a free port on it.

The frontend's port select is two selects in one field: a node, and then the free ports of that node. A
``CascadeWidget`` is that for the agent. Its ``steps`` are asked one after the other in the field's place;
each step's choice is kept on the session (``FormFillSession.steps``), never sent to core; once every step is
chosen, the field's options are what ``fetch_chosen`` returns for those choices, and the field is asked — and
resolved — like any other widget field.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator_agent.form_fill.widgets.base import Option, Widget, WidgetContext

# The options of one step: the field, the context, and the steps chosen before it.
StepOptions = Callable[[Mapping[str, Any], WidgetContext, Mapping[str, Any]], Awaitable[Sequence[Option]]]


@dataclass(frozen=True)
class Step:
    """One step before a cascade field's value: its key on the session, its title, its options."""

    key: str
    title: str
    options: StepOptions


@dataclass(frozen=True)
class Pending:
    """The step a cascade field is at: the step itself, and the choices made before it."""

    step: Step
    chosen: Mapping[str, Any]


class CascadeWidget(Widget):
    """A widget whose options depend on choices a person makes first (``steps``), in the field's place.

    A deployment widget implements ``matches``, ``steps`` and ``fetch_chosen``; ``enrich`` asks the first
    step not chosen yet, and the field's own options once every step is.
    """

    @abstractmethod
    def steps(self, field: Mapping[str, Any]) -> Sequence[Step]: ...

    @abstractmethod
    async def fetch_chosen(
        self, field: Mapping[str, Any], ctx: WidgetContext, chosen: Mapping[str, Any]
    ) -> Sequence[Option]: ...

    def pending(self, field: Mapping[str, Any], chosen: Mapping[str, Any]) -> Pending | None:
        """The first step not chosen yet, with the choices before it; None once the field itself can be asked."""
        step = next((step for step in self.steps(field) if step.key not in chosen), None)
        return None if step is None else Pending(step, dict(chosen))

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        """No steps chosen: a cascade field's options are only known once its steps are (``fetch_chosen``)."""
        return None


__all__ = ["CascadeWidget", "Pending", "Step", "StepOptions"]
