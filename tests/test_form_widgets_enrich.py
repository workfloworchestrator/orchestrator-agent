"""``enrich``: a page schema as core sent it, with what the widgets know written in."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator_agent.form_fill.model import FieldSpec, fields_in, options_of
from orchestrator_agent.form_fill.widgets import MAX_INLINE, Option, Widget, WidgetContext
from orchestrator_agent.form_fill.widgets.enrich import enrich

from .test_form_widgets import no_graphql, no_tool


def fields_of_schema(schema: dict[str, Any]) -> dict[str, FieldSpec]:
    """The enriched page as the skill reads it: one field per property, by name."""
    return {field.name: field for field in fields_in(schema)}


def values(field: FieldSpec) -> tuple[Any, ...] | None:
    """The option values a field (or its items) is limited to; None when it is free."""
    options = options_of(field)
    return None if options is None else tuple(option.value for option in options)


def labels(field: FieldSpec) -> dict[str, str]:
    return {str(option.value): option.label for option in options_of(field) or ()}


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
    fields = fields_of_schema(page.schema)
    assert values(fields["customer_id"]) == ("c-0", "c-1", "c-2") and fields["customer_id"].required
    assert labels(fields["customer_id"]) == {"c-0": "Customer 0", "c-1": "Customer 1", "c-2": "Customer 2"}
    assert values(fields["backup"]) == ("c-0", "c-1", "c-2") and not fields["backup"].required
    assert fields["customers"].kind == "list" and values(fields["customers"]) == ("c-0", "c-1", "c-2")
    assert fields["customer_id"].widget == {"id": "customerId"}
    assert fields["note"].widget is None and fields["note"].kind == "string" and fields["note"].nullable  # untouched


async def test_a_long_list_is_marked_and_kept_for_resolving():
    page = await enrich(PAGE, [Customers(MANY)], CTX)
    assert set(page.choices) == {"customer_id", "backup", "customers"}
    customer = page.choices["customer_id"]
    assert customer.options == tuple(MANY) and customer.field["format"] == "customerId"
    assert customer.title == "Customer Id" and customer.multiple is False
    customers = page.choices["customers"]
    assert customers.field == {"format": "customerId", "type": "string"} and customers.multiple is True  # its items
    fields = fields_of_schema(page.schema)
    assert fields["customer_id"].kind == "string" and values(fields["customer_id"]) is None  # asked as text
    assert fields["customer_id"].widget == {"id": "customerId", "total": 30}


@pytest.mark.parametrize(
    "count,inlined",
    [pytest.param(MAX_INLINE, True, id="ten-inlined"), pytest.param(MAX_INLINE + 1, False, id="eleven-long")],
)
async def test_the_inline_threshold(count, inlined):
    options = [Option(f"c-{n}", f"Customer {n}") for n in range(count)]
    page = await enrich(PAGE, [Customers(options)], CTX)
    mark = fields_of_schema(page.schema)["customer_id"].widget
    assert ("total" not in mark) is inlined and page.choices["customer_id"].options == tuple(options)


async def test_options_that_wait_for_another_value_mark_the_field():
    page = await enrich(PAGE, [Customers(None)], CTX)
    assert fields_of_schema(page.schema)["customer_id"].widget == {"id": "customerId", "later": True}
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


async def test_options_with_number_values_keep_their_labels():
    ports = [Option(101, "xe-0/0/1 (free) (10GBASE-LR)"), Option(102, "xe-0/0/2 (free) (10GBASE-LR)")]

    class Ports(Customers):
        id = "imsPortId"

        def matches(self, field: Mapping[str, Any]) -> bool:
            return field.get("format") == "imsPortId"

    page = {"properties": {"port_id": {"format": "imsPortId", "type": "integer"}}, "required": ["port_id"]}
    enriched = await enrich(page, [Ports(ports)], CTX)
    (port,) = fields_in(enriched.schema)
    assert values(port) == (101, 102)
    assert labels(port) == {"101": "xe-0/0/1 (free) (10GBASE-LR)", "102": "xe-0/0/2 (free) (10GBASE-LR)"}
