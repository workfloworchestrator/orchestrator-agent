"""``enrich``: a page schema as core sent it, with what the widgets know written in."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator_agent.form_fill.core_bridge import choices, is_list, labels, page_model, widget_mark
from orchestrator_agent.form_fill.widgets import MAX_INLINE, Option, Widget, WidgetContext
from orchestrator_agent.form_fill.widgets.enrich import enrich

from .test_form_widgets import no_graphql, no_tool

FEW = [Option(f"c-{n}", f"Customer {n}") for n in range(3)]
MANY = [Option(f"c-{n}", f"Customer {n}") for n in range(30)]

# The page core sends for ``customer_id: CustomerId``, ``backup: CustomerId | None = None``,
# ``customers: list[CustomerId] = []`` and ``note: str | None = None`` (pydantic-forms 2, as core 5.4 renders it).
PAGE = {
    "additionalProperties": False,
    "properties": {
        "customer_id": {"format": "customerId", "title": "Customer Id", "type": "string"},
        "backup": {
            "anyOf": [{"format": "customerId", "type": "string"}, {"type": "null"}],
            "default": None,
            "title": "Backup",
        },
        "customers": {
            "default": [],
            "items": {"format": "customerId", "type": "string"},
            "title": "Customers",
            "type": "array",
        },
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Note"},
    },
    "required": ["customer_id"],
    "title": "unknown",
    "type": "object",
}


class Customers(Widget):
    id = "customerId"

    def __init__(self, options: Sequence[Option] | None | Exception) -> None:
        self.result = options
        self.calls = 0

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("format") == "customerId"

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


CTX = WidgetContext(call_tool=no_tool, graphql=no_graphql)


async def test_a_short_list_becomes_cores_own_choice_shape():
    page = await enrich(PAGE, [Customers(FEW)], CTX)
    assert set(page.choices) == {"customer_id", "backup", "customers"}  # kept for resolving what is typed
    assert page.choices["customer_id"].options == tuple(FEW) and page.choices["customers"].multiple is True
    fields = page_model(page.schema).model_fields
    assert choices(fields["customer_id"]) == ("c-0", "c-1", "c-2") and fields["customer_id"].is_required()
    assert labels(fields["customer_id"]) == {"c-0": "Customer 0", "c-1": "Customer 1", "c-2": "Customer 2"}
    assert choices(fields["backup"]) == ("c-0", "c-1", "c-2") and not fields["backup"].is_required()
    assert is_list(fields["customers"]) and choices(fields["customers"]) == ("c-0", "c-1", "c-2")
    assert widget_mark(fields["customer_id"]) == {"id": "customerId"}
    assert widget_mark(fields["note"]) is None and fields["note"].annotation == str | None  # untouched


async def test_a_long_list_is_marked_and_kept_for_resolving():
    page = await enrich(PAGE, [Customers(MANY)], CTX)
    assert set(page.choices) == {"customer_id", "backup", "customers"}
    customer = page.choices["customer_id"]
    assert customer.options == tuple(MANY) and customer.field["format"] == "customerId"
    assert customer.title == "Customer Id" and customer.multiple is False
    customers = page.choices["customers"]
    assert customers.field == {"format": "customerId", "type": "string"} and customers.multiple is True  # its items
    fields = page_model(page.schema).model_fields
    assert fields["customer_id"].annotation is str and choices(fields["customer_id"]) is None  # asked as text
    assert widget_mark(fields["customer_id"]) == {"id": "customerId", "total": 30}


@pytest.mark.parametrize(
    "count,inlined",
    [pytest.param(MAX_INLINE, True, id="ten-inlined"), pytest.param(MAX_INLINE + 1, False, id="eleven-long")],
)
async def test_the_inline_threshold(count, inlined):
    options = [Option(f"c-{n}", f"Customer {n}") for n in range(count)]
    page = await enrich(PAGE, [Customers(options)], CTX)
    mark = widget_mark(page_model(page.schema).model_fields["customer_id"])
    assert ("total" not in mark) is inlined and page.choices["customer_id"].options == tuple(options)


async def test_options_that_wait_for_another_value_mark_the_field():
    page = await enrich(PAGE, [Customers(None)], CTX)
    assert page_model(page.schema).model_fields["customer_id"].json_schema_extra["x-widget"] == {  # type: ignore[index]
        "id": "customerId",
        "later": True,
    }
    assert page.choices == {}  # nothing to resolve against yet


async def test_a_failing_widget_leaves_the_property_as_it_was():
    page = await enrich(PAGE, [Customers(RuntimeError("core down"))], CTX)
    assert page.schema == PAGE and page.choices == {}


async def test_the_options_are_fetched_once_per_page_and_the_schema_is_not_mutated():
    widget = Customers(FEW)
    original = {**PAGE, "properties": dict(PAGE["properties"])}
    await enrich(PAGE, [widget], CTX)
    assert widget.calls == 1  # three fields of one widget with the same hints
    assert PAGE == original
