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

"""A cascade or waiting widget field in the walk: its steps, what it waits for, and when a field is taken as given.

The widgets write marks on a field (``x-widget``: ``later``, ``step``, ``stepped``) and the walk reads them here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage

from orchestrator_agent.form_fill.model import FieldSpec, fields_of
from orchestrator_agent.form_fill.verdicts import single_pick
from orchestrator_agent.form_fill.widgets.cascade import CascadeWidget
from orchestrator_agent.form_fill.widgets.enrich import LongList
from orchestrator_agent.state import FormFillSession

CHANGE_STEP = "Choose another {step}"  # the answer that goes back to a cascade field's steps (another node)


def waits(field: FieldSpec) -> bool:
    """Whether a widget field waits for another value before it has options (``later``)."""
    return bool((field.widget or {}).get("later"))


def waiting(pages: Sequence[WorkflowFormPage]) -> list[str]:
    """The widget fields of these pages that wait for another value before they have options."""
    return list(dict.fromkeys(field.name for page in pages for field in fields_of(page) if waits(field)))


def goes_back(choice: LongList, answer: Any) -> bool:
    """Whether the answer to a field asked after its steps is the choice to go back to them (another node)."""
    stepped_field = isinstance(answer, str) and choice.step is None and isinstance(choice.widget, CascadeWidget)
    return stepped_field and answer.strip().casefold().startswith(CHANGE_STEP.format(step="").strip().casefold())


def stepped(widget_choices: Mapping[str, LongList], session: FormFillSession) -> bool:
    """Whether a cascade field asked for a step now has that step chosen (so the page asks what comes next)."""
    return any(
        choice.step is not None and choice.step in session.steps.get(name, {})
        for name, choice in widget_choices.items()
    )


def takes_only_option(field: FieldSpec) -> bool:
    """Whether a field's only option is taken without asking: a required choice the form itself offers, not a widget's."""
    return field.required and single_pick(field) and field.widget is None


__all__ = ["CHANGE_STEP", "goes_back", "stepped", "takes_only_option", "waiting", "waits"]
