"""The widget layer's own parts: options, narrowing a long list, which widget a property gets."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator_agent.form_fill.widgets import (
    MAX_CANDIDATES,
    Option,
    Widget,
    WidgetContext,
    match_widget,
    narrow_options,
)

CUSTOMERS = [
    Option("c-1", "Universiteit Alder (UA)", aliases=("Universiteit Alder", "UA")),
    Option("c-2", "Universiteit Birch (UB)", aliases=("Universiteit Birch", "UB")),
    Option("c-3", "Testaccount (TA)", aliases=("Testaccount", "TA")),
]


class FormatWidget(Widget):
    """A widget matching one ``format``, with fixed options."""

    def __init__(self, id: str, fmt: str, options: Sequence[Option] | None = ()) -> None:
        self.id, self.fmt, self._options = id, fmt, options

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("format") == self.fmt

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        return self._options


async def no_tool(name: str, args: dict[str, Any]) -> Any:
    raise AssertionError(f"no tool call expected, got {name}")


async def no_fetch(url: str) -> Any:
    raise AssertionError(f"no fetch expected, got {url}")


CTX = WidgetContext(call_tool=no_tool, fetch=no_fetch)


class TestNarrowing:
    @pytest.mark.parametrize(
        "words,expected",
        [
            pytest.param("ua", ["c-1"], id="alias-token"),
            pytest.param("universiteit", ["c-1", "c-2"], id="shared-token"),
            pytest.param("universiteit birch", ["c-2", "c-1"], id="most-words-first"),
            pytest.param("birch", ["c-2"], id="label-token"),
            pytest.param("testacc", ["c-3"], id="substring"),
            pytest.param("c-3", ["c-3"], id="value"),
            pytest.param("elm", [], id="nothing"),
            pytest.param("  ", [], id="blank"),
        ],
    )
    def test_whole_words_first_then_substrings(self, words, expected):
        assert [option.value for option in narrow_options(CUSTOMERS, words)] == expected

    @pytest.mark.parametrize(
        "options,words,expected",
        [
            pytest.param(
                [Option("a", "Hogeschool-Birch"), Option("b", "Cedar Hogeschool")],
                "hogeschool birch",
                ["a", "b"],
                id="spaced-search-finds-hyphenated-name",
            ),
            pytest.param(
                [Option("a", "Hogeschool Birch"), Option("b", "Cedar Hogeschool")],
                "hogeschool-birch",
                ["a", "b"],
                id="hyphenated-search-finds-spaced-name",
            ),
            pytest.param(
                [Option("a", "Hogeschool-Birch"), Option("b", "Hogeschool Birch"), Option("c", "Cedar Hogeschool")],
                "hogeschool-birch",
                ["a", "b", "c"],
                id="same-spelling-first",
            ),
            pytest.param(
                [Option("a", "Hogeschool Birch")],
                "hogeschool-birch",
                ["a"],
                id="only-hyphenated-search",
            ),
        ],
    )
    def test_spaced_and_hyphenated_spellings_meet(self, options, words, expected):
        assert [option.value for option in narrow_options(options, words)] == expected

    def test_a_search_is_capped(self):
        many = [Option(f"c-{n}", f"Customer {n}") for n in range(120)]
        assert len(narrow_options(many, "customer")) == MAX_CANDIDATES

    async def test_options_narrows_what_fetch_returned(self):
        widget = FormatWidget("customer", "customerId", CUSTOMERS)
        assert await widget.options({}, CTX) == CUSTOMERS
        assert [o.value for o in await widget.options({}, CTX, search="alder")] == ["c-1"]
        assert await FormatWidget("later", "x", None).options({}, CTX, search="ua") is None


class TestRegistry:
    @pytest.mark.parametrize(
        "prop,matched",
        [
            pytest.param({"type": "string", "format": "customerId"}, True, id="format"),
            pytest.param({"type": "string", "format": "customerId", "enum": ["c-1"]}, False, id="core-enum-wins"),
            pytest.param({"type": "string", "format": "customerId", "const": "c-1"}, False, id="const"),
            pytest.param({"type": "string", "format": "customerId", "readOnly": True}, False, id="read-only"),
            pytest.param({"type": "string", "format": "label", "readOnly": True}, False, id="display-only"),
            pytest.param(
                {"type": "array", "items": {"type": "string", "format": "customerId"}}, True, id="array-items"
            ),
            pytest.param({"type": "string"}, False, id="plain"),
        ],
    )
    def test_matching_rules(self, prop, matched):
        widget = FormatWidget("customer", "customerId")
        assert (match_widget([widget, FormatWidget("label", "label")], prop) is widget) is matched
