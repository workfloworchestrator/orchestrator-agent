"""Typed words for a long-list field: exact names first, then the interpreter over the candidates — never free text."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from orchestrator_agent.form_fill.interpret import ModelInterpreter
from orchestrator_agent.form_fill.widgets import Option, Widget, WidgetContext
from orchestrator_agent.form_fill.widgets.enrich import LongList
from orchestrator_agent.form_fill.widgets.resolve import (
    Resolution,
    exact_options,
    resolve_answer,
    resolve_words,
    shown_as,
)

from .test_form_widgets import no_graphql, no_tool

CUSTOMERS = (
    Option("c-ut", "Universiteit Twente (UT)", aliases=("Universiteit Twente", "UT")),
    Option("c-uu", "Universiteit Utrecht (UU)", aliases=("Universiteit Utrecht", "UU")),
    Option("c-ta", "Testaccount (TA)", aliases=("Testaccount", "TA")),
    Option("c-x1", "SURF (SURF)", aliases=("SURF",)),
    Option("c-x2", "SURF (SURF2)", aliases=("SURF", "SURF2")),
    *(Option(f"c-{n}", f"Customer {n:02d}") for n in range(20)),
)
MANY = tuple(Option(f"c-{n}", f"Customer {n:03d}") for n in range(250))  # longer than the full read
FOO = Option("c-foo", "Foo, Inc.")  # a name with a comma in it
CTX = WidgetContext(call_tool=no_tool, graphql=no_graphql)


class Searchable(Widget):
    id = "customerId"

    def __init__(self, options: Sequence[Option] = CUSTOMERS) -> None:
        self._options = options
        self.searches: list[str | None] = []

    def matches(self, field: Mapping[str, Any]) -> bool:
        return True

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        return self._options

    async def options(self, field, ctx, search=None):
        self.searches.append(search)
        return await super().options(field, ctx, search)


def long_list(options: Sequence[Option] = CUSTOMERS, *, widget: Widget | None = None, multiple=False) -> LongList:
    field = {"format": "customerId", "type": "string"}
    return LongList(widget or Searchable(options), field, tuple(options), "Customer", multiple)


class FakeChooser:
    def __init__(self, picks: list[Any]) -> None:
        self.picks = picks
        self.calls: list[tuple[str, list[Any], str]] = []

    async def choose(self, title, options, words):
        self.calls.append((title, [value for value, _ in options], words))
        return self.picks


class Refuses:
    async def choose(self, title, options, words):
        raise AssertionError("no model call expected")


@pytest.mark.parametrize(
    "words,expected",
    [
        pytest.param("TESTACCOUNT", ["c-ta"], id="alias-any-case"),
        pytest.param("Testaccount (TA)", ["c-ta"], id="label"),
        pytest.param("c-ut", ["c-ut"], id="value"),
        pytest.param("surf", ["c-x1", "c-x2"], id="shared-name"),
        pytest.param("test", [], id="part-of-a-name-is-no-exact-match"),
    ],
)
def test_exact_options(words, expected):
    assert [option.value for option in exact_options(CUSTOMERS, words)] == expected


async def test_an_exact_name_needs_no_model():
    assert await resolve_words(long_list(), " testaccount ", CTX, Refuses()) == Resolution(value="c-ta", resolved=True)


async def test_exact_match_on_a_shared_name_offers_both():
    resolution = await resolve_words(long_list(), "SURF", CTX, Refuses())
    assert not resolution.resolved and [o.value for o in resolution.candidates] == ["c-x1", "c-x2"]


@pytest.mark.parametrize(
    "picks,expected",
    [
        pytest.param(["c-ut"], Resolution(value="c-ut", resolved=True), id="one-fits"),
        pytest.param(["c-ut", "c-uu"], Resolution(candidates=(CUSTOMERS[0], CUSTOMERS[1])), id="several-fit"),
        pytest.param([], Resolution(unmatched=("uni t",)), id="none-fits"),
        pytest.param(["made-up"], Resolution(unmatched=("uni t",)), id="outside-the-candidates"),
    ],
)
async def test_the_chooser_reads_the_words_over_the_candidates(picks, expected):
    chooser = FakeChooser(picks)
    assert await resolve_words(long_list(), "uni t", CTX, chooser) == expected
    ((title, values, words),) = chooser.calls
    assert title == "Customer" and words == "uni t" and values == [o.value for o in CUSTOMERS]  # all 25: a short list


async def test_a_chosen_value_outside_the_candidates_is_ignored():
    resolution = await resolve_words(long_list(), "utwente", CTX, FakeChooser(["c-ut", "c-evil"]))
    assert resolution == Resolution(value="c-ut", resolved=True)


class OwnSearch:
    """A widget whose source searches on its own (thousands of subscriptions): it implements ``options``."""

    id = "subscriptionId"

    def __init__(self, options: Sequence[Option]) -> None:
        self._options = options
        self.searches: list[str | None] = []

    def matches(self, field: Mapping[str, Any]) -> bool:
        return True

    async def options(self, field, ctx, search=None):
        self.searches.append(search)
        return self._options if search is None else self._options[:80]  # its own search: more than the cap


@pytest.mark.parametrize(
    "widget,searched",
    [
        pytest.param(Searchable(MANY), [], id="whole-list-widget-narrowed-locally"),
        pytest.param(OwnSearch(MANY), ["customer 7"], id="own-search-widget-asked"),
    ],
)
async def test_a_list_longer_than_the_full_read_is_searched_first(widget, searched):
    chooser = FakeChooser(["c-7"])
    resolution = await resolve_words(long_list(MANY, widget=widget), "customer 7", CTX, chooser)
    assert resolution == Resolution(value="c-7", resolved=True)
    assert widget.searches == searched and len(chooser.calls[0][1]) <= 50  # capped either way


@pytest.mark.parametrize(
    "words,options",
    [
        pytest.param("", CUSTOMERS, id="empty-short-list"),
        pytest.param("   ", CUSTOMERS, id="spaces-short-list"),
        pytest.param("", MANY, id="empty-searched-list"),
        pytest.param(" \t", MANY, id="spaces-searched-list"),
    ],
)
async def test_blank_words_match_nothing_without_a_search_or_a_model(words, options):
    widget = Searchable(options)
    resolution = await resolve_words(long_list(options, widget=widget), words, CTX, Refuses())
    assert resolution == Resolution(unmatched=(words,)) and widget.searches == []


class Unbounded(Searchable):
    """A widget whose search ignores the cap and returns every option."""

    async def options(self, field, ctx, search=None):
        self.searches.append(search)
        return self._options


async def test_a_search_returning_more_than_the_cap_is_cut_to_it():
    chooser = FakeChooser(["c-7"])
    resolution = await resolve_words(long_list(MANY, widget=Unbounded(MANY)), "customer 7", CTX, chooser)
    assert resolution == Resolution(value="c-7", resolved=True) and len(chooser.calls[0][1]) == 50


@pytest.mark.parametrize(
    "words,expected",
    [
        pytest.param("testaccount", Resolution(value="c-ta", resolved=True), id="exact"),
        pytest.param("customer 0", "candidates", id="a-few-are-offered"),
        pytest.param("nobody", Resolution(unmatched=("nobody",)), id="nothing"),
    ],
)
async def test_without_a_chooser_only_exact_names_resolve(words, expected):
    few = (*CUSTOMERS[:3], *(Option(f"c-{n}", f"Customer 0{n}") for n in range(3)))
    resolution = await resolve_words(long_list(few), words, CTX, None)
    if expected == "candidates":
        assert [option.label for option in resolution.candidates] == ["Customer 00", "Customer 01", "Customer 02"]
    else:
        assert resolution == expected


@pytest.mark.parametrize(
    "answer,multiple,expected",
    [
        pytest.param("c-ut", False, Resolution(value="c-ut", resolved=True), id="a-picked-value-stands"),
        pytest.param(["c-ut", "Testaccount"], True, Resolution(value=["c-ut", "c-ta"], resolved=True), id="list-items"),
        pytest.param(
            "UT, testaccount", True, Resolution(value=["c-ut", "c-ta"], resolved=True), id="list-typed-commas"
        ),
        pytest.param("UT, nobody", True, Resolution(unmatched=("nobody",)), id="list-one-item-unmatched"),
        pytest.param(
            ["UT, testaccount"], True, Resolution(value=["c-ut", "c-ta"], resolved=True), id="typed-in-a-list"
        ),
        pytest.param(
            ["c-uu", "UT, testaccount"],
            True,
            Resolution(value=["c-uu", "c-ut", "c-ta"], resolved=True),
            id="pick-and-type",
        ),
        pytest.param(
            ["Foo, Inc."], True, Resolution(value=["c-foo"], resolved=True), id="a-name-with-a-comma-in-a-list"
        ),
        pytest.param("foo, inc.", True, Resolution(value=["c-foo"], resolved=True), id="a-name-with-a-comma"),
        pytest.param("Foo, Inc.", False, Resolution(value="c-foo", resolved=True), id="one-answer-is-never-split"),
    ],
)
async def test_a_fields_answer(answer, multiple, expected):
    options = (*CUSTOMERS, FOO)
    assert await resolve_answer(long_list(options, multiple=multiple), answer, CTX, FakeChooser([])) == expected


def _scripted(picks):
    seen: dict = {}

    def model(messages, info: AgentInfo) -> ModelResponse:
        seen["prompt"] = messages[-1].parts[-1].content
        seen["schema"] = info.output_tools[0].parameters_json_schema
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args={"values": picks})])

    return FunctionModel(model), seen


async def test_the_model_interpreter_chooses_among_the_given_options_only():
    model, seen = _scripted(["c-ut"])
    options = [("c-ut", "Universiteit Twente (UT)"), ("c-uu", "Universiteit Utrecht (UU)")]
    assert await ModelInterpreter(model).choose("Customer", options, "twente") == ["c-ut"]
    assert "Universiteit Twente (UT)" in seen["prompt"] and "twente" in seen["prompt"]
    assert seen["schema"]["properties"]["values"]["items"]["enum"] == ["c-ut", "c-uu"]


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("c-ut", "Universiteit Twente (UT)", id="one"),
        pytest.param(["c-ut", "c-ta"], ["Universiteit Twente (UT)", "Testaccount (TA)"], id="list"),
        pytest.param("c-gone", "c-gone", id="unknown-shown-as-is"),
    ],
)
def test_shown_as(value, expected):
    assert shown_as(long_list(), value) == expected
