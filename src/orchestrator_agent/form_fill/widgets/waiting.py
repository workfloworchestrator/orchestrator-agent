"""A widget field that waits in the walk: its options depend on a value not known yet, and when a field is taken as given.

The widgets write a mark on a field (``x-widget``: ``later``) and the walk reads it here.
"""

from __future__ import annotations

from collections.abc import Sequence

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage

from orchestrator_agent.form_fill.model import FieldSpec, fields_of
from orchestrator_agent.form_fill.verdicts import single_pick


def waits(field: FieldSpec) -> bool:
    """Whether a widget field waits for another value before it has options (``later``)."""
    return bool((field.widget or {}).get("later"))


def waiting(pages: Sequence[WorkflowFormPage]) -> list[str]:
    """The widget fields of these pages that wait for another value before they have options."""
    return list(dict.fromkeys(field.name for page in pages for field in fields_of(page) if waits(field)))


def takes_only_option(field: FieldSpec) -> bool:
    """Whether a field's only option is taken without asking: a required choice the form itself offers, not a widget's."""
    return field.required and single_pick(field) and field.widget is None


__all__ = ["takes_only_option", "waiting", "waits"]
