"""The widget layer's own parts: options, narrowing a long list, the registry and its extender."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import sys
import types
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator_agent.form_fill.widgets import (
    MAX_CANDIDATES,
    Option,
    Widget,
    WidgetContext,
    build_widgets,
    field_hint,
    load_extender,
    match_widget,
    narrow_options,
)

CUSTOMERS = [
    Option("c-1", "Universiteit Twente (UT)", aliases=("Universiteit Twente", "UT")),
    Option("c-2", "Universiteit Utrecht (UU)", aliases=("Universiteit Utrecht", "UU")),
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


async def no_graphql(query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raise AssertionError("no GraphQL call expected")


CTX = WidgetContext(call_tool=no_tool, graphql=no_graphql)


class TestNarrowing:
    @pytest.mark.parametrize(
        "words,expected",
        [
            pytest.param("ut", ["c-1"], id="alias-token"),
            pytest.param("universiteit", ["c-1", "c-2"], id="shared-token"),
            pytest.param("universiteit utrecht", ["c-2", "c-1"], id="most-words-first"),
            pytest.param("utrecht", ["c-2"], id="label-token"),
            pytest.param("testacc", ["c-3"], id="substring"),
            pytest.param("c-3", ["c-3"], id="value"),
            pytest.param("delft", [], id="nothing"),
            pytest.param("  ", [], id="blank"),
        ],
    )
    def test_whole_words_first_then_substrings(self, words, expected):
        assert [option.value for option in narrow_options(CUSTOMERS, words)] == expected

    @pytest.mark.parametrize(
        "options,words,expected",
        [
            pytest.param(
                [Option("a", "Hogeschool-Utrecht"), Option("b", "Avans Hogeschool")],
                "hogeschool utrecht",
                ["a", "b"],
                id="spaced-search-finds-hyphenated-name",
            ),
            pytest.param(
                [Option("a", "Hogeschool Utrecht"), Option("b", "Avans Hogeschool")],
                "hogeschool-utrecht",
                ["a", "b"],
                id="hyphenated-search-finds-spaced-name",
            ),
            pytest.param(
                [Option("a", "Hogeschool-Utrecht"), Option("b", "Hogeschool Utrecht"), Option("c", "Avans Hogeschool")],
                "hogeschool-utrecht",
                ["a", "b", "c"],
                id="same-spelling-first",
            ),
            pytest.param(
                [Option("a", "Hogeschool Utrecht")],
                "hogeschool-utrecht",
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
        assert [o.value for o in await widget.options({}, CTX, search="twente")] == ["c-1"]
        assert await FormatWidget("later", "x", None).options({}, CTX, search="ut") is None


@pytest.mark.parametrize(
    "field,expected",
    [
        pytest.param({"extraProperties": {"productIds": ["a"]}}, ["a"], id="extra-properties"),
        pytest.param({"uniforms": {"productIds": ["b"]}}, ["b"], id="uniforms"),
        pytest.param(
            {"extraProperties": {"productIds": ["a"]}, "uniforms": {"productIds": ["b"]}}, ["a"], id="extra-wins"
        ),
        pytest.param({}, None, id="none"),
    ],
)
def test_a_hint_comes_from_extra_properties_then_uniforms(field, expected):
    assert field_hint(field, "productIds") == expected


class TestRegistry:
    def test_the_builtins_without_an_extender(self):
        builtin = FormatWidget("customer", "customerId")
        assert build_widgets([builtin]) == [builtin]

    def test_an_extender_prepends_and_drops_by_id(self):
        builtin, other = FormatWidget("customer", "customerId"), FormatWidget("product", "productId")
        own = FormatWidget("surf-customer", "customerId")

        def extend(widgets):
            return [own, *(widget for widget in widgets if widget.id != "product")]

        widgets = build_widgets([builtin, other], extend)
        assert [widget.id for widget in widgets] == ["surf-customer", "customer"]
        assert match_widget(widgets, {"type": "string", "format": "customerId"}) is own  # the first match wins

    def test_an_extender_must_return_widgets(self):
        with pytest.raises(ValueError, match="FORM_WIDGET_EXTENDER"):
            build_widgets([FormatWidget("customer", "customerId")], lambda widgets: None)

    @pytest.mark.parametrize(
        "prop,matched",
        [
            pytest.param({"type": "string", "format": "customerId"}, True, id="format"),
            pytest.param({"type": "string", "format": "customerId", "enum": ["c-1"]}, False, id="core-enum-wins"),
            pytest.param({"type": "string", "format": "customerId", "const": "c-1"}, False, id="const"),
            pytest.param(
                {"type": "string", "format": "customerId", "uniforms": {"disabled": True}}, False, id="read-only"
            ),
            pytest.param({"type": "string", "format": "label"}, False, id="display-only"),
            pytest.param(
                {"type": "array", "items": {"type": "string", "format": "customerId"}}, True, id="array-items"
            ),
            pytest.param({"type": "string"}, False, id="plain"),
        ],
    )
    def test_matching_rules(self, prop, matched):
        widget = FormatWidget("customer", "customerId")
        assert (match_widget([widget, FormatWidget("label", "label")], prop) is widget) is matched


class TestLoadExtender:
    def test_unset_is_none(self):
        assert load_extender(None) is None and load_extender("") is None

    def test_a_module_callable(self, monkeypatch):
        module = types.ModuleType("fake_widgets")
        module.extend = lambda widgets: widgets
        monkeypatch.setitem(sys.modules, "fake_widgets", module)
        assert load_extender("fake_widgets:extend") is module.extend

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("no_such_module_xyz:extend", id="no-module"),
            pytest.param("fake_widgets:missing", id="no-attribute"),
            pytest.param("fake_widgets:not_callable", id="not-callable"),
            pytest.param("fake_widgets", id="no-colon"),
        ],
    )
    def test_a_bad_path_fails_loudly(self, monkeypatch, path):
        module = types.ModuleType("fake_widgets")
        module.not_callable = 42
        monkeypatch.setitem(sys.modules, "fake_widgets", module)
        with pytest.raises(ValueError, match="FORM_WIDGET_EXTENDER"):
            load_extender(path)
