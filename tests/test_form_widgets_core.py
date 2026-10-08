"""The widgets for the formats orchestrator-core defines, against fake channels to core."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping
from typing import Any

import pytest

from orchestrator_agent.form_fill.widgets import Option, WidgetContext
from orchestrator_agent.form_fill.widgets.core import (
    BUILTIN_WIDGETS,
    CUSTOMERS_QUERY,
    CustomerIdWidget,
    ProductIdWidget,
)
from orchestrator_agent.tool_names import ALL_TOOL_NAMES, LIST_PRODUCTS_TOOL

CUSTOMERS = {
    "customers": {
        "page": [
            {"customerId": "c-1", "fullname": "Universiteit Twente", "shortcode": "UT"},
            {"customerId": "c-2", "fullname": "Testaccount", "shortcode": ""},
        ]
    }
}
P1, P2, P3 = (f"d8a3f9b0-0000-4000-8000-00000000000{n}" for n in (1, 2, 3))
PRODUCTS = [
    {"product_id": P1, "name": "Service Port 10G", "tag": "SP"},
    {"product_id": P2, "name": "Service Port 1G", "tag": "SP"},
    {"product_id": P3, "name": "Node", "tag": "NODE"},
]


class Channels:
    """Core's two APIs as a widget sees them, recording what was asked."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, Mapping[str, Any] | None]] = []
        self.tools: list[tuple[str, dict[str, Any]]] = []

    async def graphql(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.queries.append((query, variables))
        return CUSTOMERS

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.tools.append((name, args))
        return PRODUCTS

    def context(self) -> WidgetContext:
        return WidgetContext(call_tool=self.call_tool, graphql=self.graphql)


def test_the_builtins_are_the_formats_core_defines():
    assert [widget.id for widget in BUILTIN_WIDGETS] == ["customerId", "productId"]
    assert LIST_PRODUCTS_TOOL == "list_products" and LIST_PRODUCTS_TOOL in ALL_TOOL_NAMES


@pytest.mark.parametrize(
    "widget,field,matched",
    [
        pytest.param(CustomerIdWidget(), {"type": "string", "format": "customerId"}, True, id="customer"),
        pytest.param(CustomerIdWidget(), {"type": "integer", "format": "customerId"}, False, id="customer-not-string"),
        pytest.param(ProductIdWidget(), {"type": "string", "format": "productId"}, True, id="product"),
        pytest.param(ProductIdWidget(), {"type": "string", "format": "uuid"}, False, id="other-format"),
    ],
)
def test_matching(widget, field, matched):
    assert widget.matches(field) is matched


async def test_customers_come_from_core_graphql_labelled_with_their_shortcode():
    channels = Channels()
    options = await CustomerIdWidget().options({"type": "string", "format": "customerId"}, channels.context())
    assert options == [
        Option("c-1", "Universiteit Twente (UT)", aliases=("Universiteit Twente", "UT")),
        Option("c-2", "Testaccount", aliases=("Testaccount",)),
    ]
    ((query, _),) = channels.queries
    assert query == CUSTOMERS_QUERY


@pytest.mark.parametrize(
    "field,expected",
    [
        pytest.param({"format": "productId"}, [P1, P2, P3], id="all"),
        pytest.param({"format": "productId", "extraProperties": {"productIds": [P2, P1]}}, [P1, P2], id="restricted"),
        pytest.param({"format": "productId", "uniforms": {"productIds": [P3]}}, [P3], id="restricted-uniforms"),
    ],
)
async def test_products_come_from_list_products(field, expected):
    channels = Channels()
    options = await ProductIdWidget().options(field, channels.context())
    assert [option.value for option in options or []] == expected
    assert channels.tools == [(LIST_PRODUCTS_TOOL, {})]
    assert options and options[0].label == {P1: "Service Port 10G", P3: "Node"}[expected[0]]
