# Form Field Widgets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The form-fill skill resolves the options of pydantic-forms fields whose options are not in core's schema (`format: customerId`, `format: productId`, and deployment formats through an extender), so they are asked as chips or resolved from typed words to one real option — never sent to core as free text.

**Architecture:** A widget registry mirroring pydantic-forms' `ComponentMatcherExtender`: each `FieldWidget` matches a property schema and fetches its `Option`s from core (MCP or GraphQL, with the caller's token). `enrich()` rewrites a page schema before the page model is built: a short list becomes core's own `enum` + `options` shape, a long list gets an `x-widget` mark and its typed answer is resolved (exact match, then the interpreter over candidates) before the page is submitted. Everything downstream (chips, labels, approval) works from the rewritten schema.

**Tech Stack:** Python 3.11–3.13, pydantic 2.13, pydantic-ai 1.56 (`MCPToolset`, `FunctionModel` in tests), httpx, orchestrator-core 5.4.0 (MCP + GraphQL), pytest + pytest-asyncio (`asyncio_mode=auto`), docker compose for the integration stack.

**Spec:** `docs/superpowers/specs/2026-10-07-form-field-widgets-design.md`

## Global Constraints

- New `src/` files start with the repo's Apache license header (`# Copyright 2019-2026 SURF, GÉANT.` + the 10 license lines, as in `src/orchestrator_agent/form_fill/skill.py`) and `from __future__ import annotations`.
- ruff (line length 120, google docstrings) and `ruff format` must pass; run `uv run ruff check . && uv run ruff format --check .`.
- Do not run mypy per task (slow); commit with `PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit ...`. Commit messages carry no `Co-Authored-By` line.
- Python style: prefer comprehensions / `itertools` / `next(..., None)` over loops with `break`/`continue`; `match`/`case` over `isinstance` chains; tests that differ only in data use `@pytest.mark.parametrize` with `pytest.param(..., id=...)`.
- Test modules set `os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")` before importing `orchestrator_agent` (as every existing test does).
- No new runtime dependency (httpx is already one).
- Thresholds, verbatim from the spec: at most **10** options are inlined as chips (`MAX_INLINE = 10`); a long list of at most **200** options is read in full by the interpreter (`MAX_FULL_READ = 200`); a search returns at most **50** candidates (`MAX_CANDIDATES = 50`).
- The schema mark is the property key `"x-widget"`; settings are `FORM_WIDGET_EXTENDER` (`"package.module:callable"`, unset by default) and `WFO_CORE_GRAPHQL_URL` (unset: derived from `WFO_CORE_MCP_URL`, `…/mcp` → `…/api/graphql`).
- Built-in widgets cover only the formats orchestrator-core defines: `customerId`, `productId`. Nothing SURF-specific is added to this repo.
- Integration tests run against `ghcr.io/workfloworchestrator/orchestrator-core:5.4.0` and are skipped unless `WFO_INTEGRATION_CORE_URL` is set.

## Review Focus

1. **Typed words for a widget field that resolve to no single option** must never reach core (core's `CustomerId` accepts any string) — the page stops and asks again. Pinned in Task 6 (`test_unresolved_words_are_never_submitted`).
2. **The widget's source fails** (GraphQL unreachable, an MCP tool error) — the field is asked as free text exactly as today and the form stays open. Pinned in Task 4 (`test_a_failing_widget_leaves_the_property_as_it_was`) and Task 8 (`test_unreachable_graphql_leaves_customer_free_text`).
3. **The interpreter names a value that is not one of the candidates** — ignored, never taken. Pinned in Task 5 (`test_a_chosen_value_outside_the_candidates_is_ignored`).
4. **Two options with the same label or alias** (two customers called the same) — the exact match yields both and the person picks; it is never "first one wins". Pinned in Task 5 (`test_exact_match_on_a_shared_name_offers_both`).
5. **A session persisted before this change** (no `resolved`, `AskField` without `hint` in `pending`) — still loads and continues. Pinned in Task 6 (`test_a_session_persisted_before_widgets_still_loads`).

## File Structure

| File | Responsibility |
|---|---|
| `src/orchestrator_agent/form_fill/widgets/__init__.py` | Public surface of the widget layer (re-exports) |
| `src/orchestrator_agent/form_fill/widgets/base.py` | `Option`, `WidgetContext`, `FieldWidget` protocol, `Widget` base class, `narrow_options`, `field_hint`, constants |
| `src/orchestrator_agent/form_fill/widgets/registry.py` | `build_widgets`, `load_extender`, `match_widget` |
| `src/orchestrator_agent/form_fill/widgets/graphql.py` | `GraphQL` protocol, `CoreGraphQL` client, `core_auth`, `graphql_url` |
| `src/orchestrator_agent/form_fill/widgets/core.py` | Built-ins: `CustomerIdWidget`, `ProductIdWidget`, `BUILTIN_WIDGETS` |
| `src/orchestrator_agent/form_fill/widgets/enrich.py` | `enrich` (schema rewrite), `EnrichedPage`, `LongList` |
| `src/orchestrator_agent/form_fill/widgets/resolve.py` | `exact_options`, `resolve_words`, `resolve_answer`, `shown_as`, `Resolution` |
| `src/orchestrator_agent/form_fill/core_bridge.py` | Carry `x-widget` into the field; `widget_mark` |
| `src/orchestrator_agent/form_fill/interpret.py` | `Chooser` protocol, `ModelInterpreter.choose` |
| `src/orchestrator_agent/form_fill/skill.py` | Enrich + resolve in `_walk`; hints in `question()`; skip `later` fields |
| `src/orchestrator_agent/form_fill/__init__.py` | `build_form_fill_skill` wires widgets, GraphQL, chooser |
| `src/orchestrator_agent/state.py` | `AskField.hint`, `FormFillSession.resolved` |
| `src/orchestrator_agent/adapters/chat/librechat.py` | `hint` into the question's description |
| `src/orchestrator_agent/adapters/a2a/kagent.py` | `hint` after the question text |
| `src/orchestrator_agent/app.py` | Build the skill once; log its widgets |
| `src/orchestrator_agent/settings.py` | `FORM_WIDGET_EXTENDER`, `WFO_CORE_GRAPHQL_URL` |
| `src/orchestrator_agent/tool_names.py` | `LIST_PRODUCTS_TOOL` (in `ALL_TOOL_NAMES`) |
| `pyproject.toml` | Register the `integration` pytest marker |
| `tests/test_form_widgets.py` | Base types, narrowing, registry, extender loading |
| `tests/test_form_widgets_graphql.py` | GraphQL client, auth forwarding, URL derivation |
| `tests/test_form_widgets_core.py` | Built-in widgets against fake channels |
| `tests/test_form_widgets_enrich.py` | `enrich` and the `core_bridge` mark |
| `tests/test_form_widgets_resolve.py` | Resolution and `ModelInterpreter.choose` |
| `tests/test_form_widgets_skill.py` | The skill walking a widget page; hints in both transports; building the skill |
| `tests/integration/` | Compose stack, core app (customers, products, `widget_demo` task), integration tests |
| `.github/workflows/ci.yml` | `integration` job |
| `README.md` | Settings and the extender |

---

### Task 1: Widget types, narrowing and the registry

**Files:**
- Create: `src/orchestrator_agent/form_fill/widgets/__init__.py`
- Create: `src/orchestrator_agent/form_fill/widgets/base.py`
- Create: `src/orchestrator_agent/form_fill/widgets/registry.py`
- Modify: `src/orchestrator_agent/settings.py` (add `FORM_WIDGET_EXTENDER`)
- Test: `tests/test_form_widgets.py`

**Interfaces:**
- Consumes: `orchestrator_agent.form_fill.skill.CallTool` (existing protocol: `async (name: str, args: dict) -> Any`); `core_bridge.DISPLAY_ONLY_FORMATS`, `FORMAT`, `is_read_only` (existing).
- Produces:
  - `Option(value: str | int, label: str, aliases: tuple[str, ...] = ())` — frozen dataclass.
  - `GraphQL` protocol: `async __call__(query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]` (implemented in Task 2).
  - `WidgetContext(call_tool: CallTool, graphql: GraphQL, values: Mapping[str, Any] = {})` — frozen dataclass.
  - `FieldWidget` protocol: `id: str`; `matches(field) -> bool`; `async options(field, ctx, search: str | None = None) -> Sequence[Option] | None`.
  - `Widget` abstract base: subclasses set `id`, implement `matches` and `async fetch(field, ctx) -> Sequence[Option] | None`; `options()` narrows `fetch` with `narrow_options`.
  - `narrow_options(options, words, limit=MAX_CANDIDATES) -> list[Option]`; `field_hint(field, key) -> Any`.
  - Constants `MAX_INLINE = 10`, `MAX_FULL_READ = 200`, `MAX_CANDIDATES = 50`, `WIDGET_MARK = "x-widget"`.
  - `FieldWidgetExtender = Callable[[list[FieldWidget]], list[FieldWidget]]`.
  - `build_widgets(builtins, extender=None) -> list[FieldWidget]` (raises `ValueError` when the extender returns no list of widgets).
  - `load_extender(path: str | None) -> FieldWidgetExtender | None` (raises `ValueError` mentioning `FORM_WIDGET_EXTENDER`).
  - `match_widget(widgets, prop) -> FieldWidget | None` — applies the matching rules.

- [ ] **Step 1: Write the failing tests** — create `tests/test_form_widgets.py`:

```python
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
            build_widgets([FormatWidget("customer", "customerId")], lambda widgets: None)  # type: ignore[arg-type,return-value]

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
        module.extend = lambda widgets: widgets  # type: ignore[attr-defined]
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
        module.not_callable = 42  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_widgets", module)
        with pytest.raises(ValueError, match="FORM_WIDGET_EXTENDER"):
            load_extender(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'orchestrator_agent.form_fill.widgets'`.

- [ ] **Step 3: Write `src/orchestrator_agent/form_fill/widgets/base.py`** (license header first):

```python
"""What a widget is: the agent's counterpart of a pydantic-forms component matcher.

The browser renders a field with the first component whose matcher accepts it, and a component such as the
customer select fetches its own options. Core's page schema carries only the ``format`` marker and hints for
those fields; a widget is what the agent knows about one such format — whether a property is one
(``matches``) and the options it has (``options``), fetched from core as the person asking.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from orchestrator_agent.form_fill.skill import CallTool

MAX_INLINE = 10  # at most this many options are asked as chips (LibreChat takes twelve, "Keep default" included)
MAX_FULL_READ = 200  # a long list this short is read in full by the interpreter, so abbreviations still resolve
MAX_CANDIDATES = 50  # a search returns at most this many options
WIDGET_MARK = "x-widget"  # the property key ``enrich`` marks a widget field with


@dataclass(frozen=True)
class Option:
    """One option of a widget field: the value sent to core, how a person sees it, other names they may type."""

    value: str | int
    label: str
    aliases: tuple[str, ...] = ()


class GraphQL(Protocol):
    """A query against core's GraphQL API as the person asking; the ``data`` of the answer."""

    async def __call__(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class WidgetContext:
    """What a widget may use to find its options: core's two APIs, and the form values known so far."""

    call_tool: CallTool
    graphql: GraphQL
    values: Mapping[str, Any] = field(default_factory=dict)


class FieldWidget(Protocol):
    """A format the agent knows: whether a property is one, and its options.

    ``options(search=None)`` is every option; with ``search``, at most ``MAX_CANDIDATES`` the words could
    mean. None means the options depend on a value not known yet: the field waits for it.
    """

    id: str

    def matches(self, field: Mapping[str, Any]) -> bool: ...

    async def options(
        self, field: Mapping[str, Any], ctx: WidgetContext, search: str | None = None
    ) -> Sequence[Option] | None: ...


FieldWidgetExtender = Callable[[list[FieldWidget]], list[FieldWidget]]


class Widget(ABC):
    """A widget whose source returns the whole list: a search narrows what ``fetch`` returned.

    A widget whose source can search on its own (thousands of subscriptions) implements ``options`` instead.
    """

    id: str

    @abstractmethod
    def matches(self, field: Mapping[str, Any]) -> bool: ...

    @abstractmethod
    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None: ...

    async def options(
        self, field: Mapping[str, Any], ctx: WidgetContext, search: str | None = None
    ) -> Sequence[Option] | None:
        fetched = await self.fetch(field, ctx)
        if fetched is None or search is None:
            return fetched
        return narrow_options(fetched, search)


def names_of(option: Option) -> tuple[str, ...]:
    """Every name a person may know an option by: its label, its value, its aliases."""
    return (option.label, str(option.value), *option.aliases)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _shared_words(option: Option, wanted: set[str]) -> int:
    return len(wanted & set().union(*map(_words, names_of(option))))


def narrow_options(options: Sequence[Option], words: str, limit: int = MAX_CANDIDATES) -> list[Option]:
    """The options the words could mean: those sharing the most whole words first, then those containing the words."""
    needle = words.strip().casefold()
    if not needle:
        return []
    wanted = _words(words)
    shared = {id(option): _shared_words(option, wanted) for option in options}
    by_word = sorted((o for o in options if shared[id(o)]), key=lambda o: -shared[id(o)])
    by_text = [o for o in options if not shared[id(o)] and any(needle in name.casefold() for name in names_of(o))]
    return [*by_word, *by_text][:limit]


def field_hint(field: Mapping[str, Any], key: str) -> Any:
    """A hint the form put on the field for its component (``extraProperties``, else the older ``uniforms``)."""
    places = (field.get("extraProperties"), field.get("uniforms"))
    return next((hints[key] for hints in places if isinstance(hints, Mapping) and key in hints), None)


__all__ = [
    "MAX_CANDIDATES",
    "MAX_FULL_READ",
    "MAX_INLINE",
    "WIDGET_MARK",
    "FieldWidget",
    "FieldWidgetExtender",
    "GraphQL",
    "Option",
    "Widget",
    "WidgetContext",
    "field_hint",
    "names_of",
    "narrow_options",
]
```

- [ ] **Step 4: Write `src/orchestrator_agent/form_fill/widgets/registry.py`** (license header first):

```python
"""Which widget a property is: the built-ins, as a deployment's extender rearranges them.

Mirrors pydantic-forms' ``ComponentMatcherExtender``: the extender receives the agent's widgets and returns
the list to use; the first widget that matches a property is its widget, so an extender puts its own
first or drops a built-in by ``id``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from orchestrator_agent.form_fill.core_bridge import DISPLAY_ONLY_FORMATS, FORMAT, is_read_only
from orchestrator_agent.form_fill.widgets.base import FieldWidget, FieldWidgetExtender

SETTING = "FORM_WIDGET_EXTENDER"


def build_widgets(builtins: Sequence[FieldWidget], extender: FieldWidgetExtender | None = None) -> list[FieldWidget]:
    """The widgets in the order they are tried: the built-ins, or what the extender made of them."""
    if extender is None:
        return list(builtins)
    widgets = extender(list(builtins))
    if not isinstance(widgets, list) or not all(hasattr(widget, "matches") for widget in widgets):
        raise ValueError(f"{SETTING}: the extender must return a list of widgets, got {widgets!r}")
    return widgets


def load_extender(path: str | None) -> FieldWidgetExtender | None:
    """The extender named ``package.module:callable``; None when unset. A bad path fails loudly, at startup."""
    if not path:
        return None
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"{SETTING}={path!r}: expected 'package.module:callable'")
    try:
        extender = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{SETTING}={path!r}: {exc}") from exc
    if not callable(extender):
        raise ValueError(f"{SETTING}={path!r}: {attribute} is not callable")
    return extender  # type: ignore[no-any-return]


def widget_target(prop: Mapping[str, Any]) -> Mapping[str, Any]:
    """What a widget is matched on: a list's items, any other property itself."""
    items = prop.get("items")
    return items if prop.get("type") == "array" and isinstance(items, Mapping) else prop


def match_widget(widgets: Sequence[FieldWidget], prop: Mapping[str, Any]) -> FieldWidget | None:
    """The widget of a resolved property, if any.

    Core's own options always win (a property with ``enum`` or ``const``), what nobody fills is no widget's
    (read-only or display-only), and a list is matched on its items.
    """
    target = widget_target(prop)
    if any(key in target for key in ("enum", "const")) or is_read_only(prop) or is_read_only(target):
        return None
    if target.get(FORMAT) in DISPLAY_ONLY_FORMATS:
        return None
    return next((widget for widget in widgets if widget.matches(target)), None)


__all__ = ["SETTING", "build_widgets", "load_extender", "match_widget", "widget_target"]
```

- [ ] **Step 5: Write `src/orchestrator_agent/form_fill/widgets/__init__.py`** (license header first):

```python
"""Widgets: what the agent knows about the form fields whose options core leaves to the frontend."""

from orchestrator_agent.form_fill.widgets.base import (
    MAX_CANDIDATES,
    MAX_FULL_READ,
    MAX_INLINE,
    WIDGET_MARK,
    FieldWidget,
    FieldWidgetExtender,
    GraphQL,
    Option,
    Widget,
    WidgetContext,
    field_hint,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.registry import build_widgets, load_extender, match_widget, widget_target

__all__ = [
    "MAX_CANDIDATES",
    "MAX_FULL_READ",
    "MAX_INLINE",
    "WIDGET_MARK",
    "FieldWidget",
    "FieldWidgetExtender",
    "GraphQL",
    "Option",
    "Widget",
    "WidgetContext",
    "build_widgets",
    "field_hint",
    "load_extender",
    "match_widget",
    "names_of",
    "narrow_options",
    "widget_target",
]
```

- [ ] **Step 6: Add the setting** — in `src/orchestrator_agent/settings.py`, after `AGENT_DOMAIN_CONTEXT`:

```python
    FORM_WIDGET_EXTENDER: str | None = Field(
        default=None,
        description="Optional 'package.module:callable' that receives the form-fill widgets and returns the list "
        "to use (a deployment's own formats first). Unset uses the built-ins (customerId, productId).",
    )
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form_widgets.py -q`
Expected: all pass.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/settings.py tests/test_form_widgets.py
uv run ruff format src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/settings.py tests/test_form_widgets.py
git add src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/settings.py tests/test_form_widgets.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Add form-fill widget types, narrowing and registry"
```

---

### Task 2: Core's GraphQL API as the person asking

**Files:**
- Create: `src/orchestrator_agent/form_fill/widgets/graphql.py`
- Modify: `src/orchestrator_agent/form_fill/widgets/__init__.py` (re-export)
- Modify: `src/orchestrator_agent/settings.py` (add `WFO_CORE_GRAPHQL_URL`)
- Test: `tests/test_form_widgets_graphql.py`

**Interfaces:**
- Consumes: `orchestrator_agent.mcp_client._ContextVarBearerAuth`, `bind_outbound_token` (existing); `GraphQL` protocol (Task 1).
- Produces:
  - `graphql_url(mcp_url: str, configured: str | None = None) -> str` — `configured`, else `mcp_url` with a trailing `/mcp` (or `/mcp/`) replaced by `/api/graphql`.
  - `core_auth() -> httpx.Auth` — the MCP client's auth (forwarded user token, else the refreshed service token).
  - `GraphQLError(Exception)`.
  - `CoreGraphQL(url: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0)` — implements `GraphQL`: POSTs `{"query", "variables"}`, raises `GraphQLError` on an `errors` list or a missing `data`, `httpx.HTTPStatusError` on a non-2xx status.

- [ ] **Step 1: Write the failing tests** — create `tests/test_form_widgets_graphql.py`:

```python
"""Core's GraphQL API, called as the person asking: the URL, the token, the answer or its errors."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from orchestrator_agent.auth import token_manager
from orchestrator_agent.form_fill.widgets.graphql import CoreGraphQL, GraphQLError, graphql_url
from orchestrator_agent.mcp_client import bind_outbound_token

URL = "https://core.example.com/api/graphql"


@pytest.mark.parametrize(
    "mcp_url,configured,expected",
    [
        pytest.param("http://core:8080/mcp", None, "http://core:8080/api/graphql", id="derived"),
        pytest.param("http://core:8080/mcp/", None, "http://core:8080/api/graphql", id="trailing-slash"),
        pytest.param("http://core:8080/mcp", "http://gql/x", "http://gql/x", id="configured-wins"),
    ],
)
def test_the_url_follows_the_mcp_url_unless_configured(mcp_url, configured, expected):
    assert graphql_url(mcp_url, configured) == expected


def _client(handler) -> tuple[CoreGraphQL, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, len(seen))

    return CoreGraphQL(URL, transport=httpx.MockTransport(record)), seen


@pytest.fixture(autouse=True)
def no_service_token(monkeypatch):
    """No outbound OAuth2 unless a test turns it on: nothing may go looking for a token URL."""
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)


async def test_the_query_goes_out_with_the_callers_token_and_its_data_comes_back():
    client, seen = _client(lambda request, n: httpx.Response(200, json={"data": {"customers": {"page": []}}}))
    with bind_outbound_token("user-token"):
        assert await client("query { customers { page { customerId } } }", {"first": 5}) == {"customers": {"page": []}}
    (request,) = seen
    assert request.headers["authorization"] == "Bearer user-token"
    assert json.loads(request.content) == {"query": "query { customers { page { customerId } } }", "variables": {"first": 5}}


async def test_the_service_token_is_refreshed_once_on_401(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", True)
    monkeypatch.setattr(token_manager, "get_token", AsyncMock(return_value="old"))
    monkeypatch.setattr(token_manager, "refresh_token", AsyncMock(return_value="new"))
    client, seen = _client(lambda request, n: httpx.Response(401 if n == 1 else 200, json={"data": {}}))
    assert await client("query { version { applicationVersions } }") == {}
    assert [request.headers["authorization"] for request in seen] == ["Bearer old", "Bearer new"]


@pytest.mark.parametrize(
    "response,error",
    [
        pytest.param(httpx.Response(200, json={"errors": [{"message": "Not authorized"}]}), GraphQLError, id="errors"),
        pytest.param(httpx.Response(200, json={}), GraphQLError, id="no-data"),
        pytest.param(httpx.Response(500, text="boom"), httpx.HTTPStatusError, id="status"),
    ],
)
async def test_a_failed_answer_raises(response, error):
    client, _ = _client(lambda request, n: response)
    with pytest.raises(error):
        await client("query { customers { page { customerId } } }")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets_graphql.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'orchestrator_agent.form_fill.widgets.graphql'`.

- [ ] **Step 3: Write `src/orchestrator_agent/form_fill/widgets/graphql.py`** (license header first):

```python
"""Core's GraphQL API, as the person asking: where the frontend's components get most of their options.

Customers have no MCP tool in core; the frontend reads them over GraphQL, and so does a widget. The request
carries the same token as the agent's MCP calls (``core_auth``), so a widget sees what the person may see.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from orchestrator_agent.mcp_client import _ContextVarBearerAuth


class GraphQLError(Exception):
    """Core answered the query with errors, or without data."""


def graphql_url(mcp_url: str, configured: str | None = None) -> str:
    """Core's GraphQL endpoint: as configured, else beside its MCP endpoint (``…/mcp`` -> ``…/api/graphql``)."""
    if configured:
        return configured
    base = mcp_url.rstrip("/")
    return (base.removesuffix("/mcp") if base.endswith("/mcp") else base) + "/api/graphql"


def core_auth() -> httpx.Auth:
    """The auth of the agent's calls to core: the caller's forwarded token, else the service token.

    For a deployment's widget that reads its own endpoints of core with an ``httpx.AsyncClient``.
    """
    return _ContextVarBearerAuth()


class CoreGraphQL:
    """A ``GraphQL`` (``widgets.base``) over HTTP, authenticated like the agent's MCP calls."""

    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0) -> None:
        self.url, self.transport, self.timeout = url, transport, timeout

    async def __call__(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(auth=core_auth(), transport=self.transport, timeout=self.timeout) as client:
            response = await client.post(self.url, json={"query": query, "variables": dict(variables or {})})
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise GraphQLError("; ".join(str(error.get("message", error)) for error in body["errors"]))
        if not isinstance(body.get("data"), dict):
            raise GraphQLError(f"no data in the answer of {self.url}")
        return body["data"]  # type: ignore[no-any-return]


__all__ = ["CoreGraphQL", "GraphQLError", "core_auth", "graphql_url"]
```

- [ ] **Step 4: Re-export** — in `src/orchestrator_agent/form_fill/widgets/__init__.py` add
`from orchestrator_agent.form_fill.widgets.graphql import CoreGraphQL, GraphQLError, core_auth, graphql_url`
and the four names to `__all__` (alphabetical order).

- [ ] **Step 5: Add the setting** — in `src/orchestrator_agent/settings.py`, after `WFO_CORE_MCP_URL`:

```python
    WFO_CORE_GRAPHQL_URL: str | None = Field(
        default=None,
        description="URL of orchestrator-core's GraphQL API, which form-fill widgets read options from (customers). "
        "Unset: derived from WFO_CORE_MCP_URL ('…/mcp' -> '…/api/graphql').",
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form_widgets_graphql.py tests/test_mcp_client.py -q`
Expected: all pass.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/settings.py tests/test_form_widgets_graphql.py
uv run ruff format src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/settings.py tests/test_form_widgets_graphql.py
git add src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/settings.py tests/test_form_widgets_graphql.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Add a GraphQL client to core for form-fill widgets"
```

---

### Task 3: The built-in widgets — `customerId` and `productId`

**Files:**
- Create: `src/orchestrator_agent/form_fill/widgets/core.py`
- Modify: `src/orchestrator_agent/form_fill/widgets/__init__.py` (re-export)
- Modify: `src/orchestrator_agent/tool_names.py` (add `LIST_PRODUCTS_TOOL`)
- Test: `tests/test_form_widgets_core.py`

**Interfaces:**
- Consumes: `Widget`, `Option`, `WidgetContext`, `field_hint` (Task 1); `GraphQL` (Task 1/2).
- Produces:
  - `LIST_PRODUCTS_TOOL = "list_products"` in `tool_names.py`, added to `ALL_TOOL_NAMES` and `__all__` (core 5.4.0 exposes it; the startup contract check then covers it).
  - `CUSTOMERS_QUERY: str`; `CustomerIdWidget()` with `id = "customerId"`; `ProductIdWidget()` with `id = "productId"`.
  - `BUILTIN_WIDGETS: tuple[FieldWidget, ...] = (CustomerIdWidget(), ProductIdWidget())`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_form_widgets_core.py`:

```python
"""The widgets for the formats orchestrator-core defines, against fake channels to core."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping
from typing import Any

import pytest

from orchestrator_agent.form_fill.widgets import Option, WidgetContext
from orchestrator_agent.form_fill.widgets.core import BUILTIN_WIDGETS, CUSTOMERS_QUERY, CustomerIdWidget, ProductIdWidget
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets_core.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'orchestrator_agent.form_fill.widgets.core'`.

- [ ] **Step 3: Add the tool name** — in `src/orchestrator_agent/tool_names.py`, under the form-fill block:

```python
# The options of a ``productId`` form field (form-fill widgets).
LIST_PRODUCTS_TOOL = "list_products"
```

add `LIST_PRODUCTS_TOOL,` to `ALL_TOOL_NAMES` (after `SUBSCRIPTION_WORKFLOWS_TOOL`) and `"LIST_PRODUCTS_TOOL",` to `__all__`.

- [ ] **Step 4: Write `src/orchestrator_agent/form_fill/widgets/core.py`** (license header first):

```python
"""The widgets for the formats orchestrator-core itself defines: ``customerId`` and ``productId``.

Every other format a form may carry (subscriptions, ports, contacts, locations, ...) belongs to a deployment
and comes with its extender.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, TypeAdapter

from orchestrator_agent.form_fill.widgets.base import FieldWidget, Option, Widget, WidgetContext, field_hint
from orchestrator_agent.tool_names import LIST_PRODUCTS_TOOL

# The frontend's customer select asks the same (``useGetCustomersQuery``): every customer, once.
CUSTOMERS_QUERY = "query Customers { customers(first: 1000000, after: 0) { page { customerId fullname shortcode } } }"


class _Customer(BaseModel):
    customerId: str
    fullname: str
    shortcode: str = ""


class _Product(BaseModel):
    """What a widget needs of a row of ``list_products`` (core's ``ProductSchema``)."""

    product_id: str
    name: str


_CUSTOMERS: TypeAdapter[list[_Customer]] = TypeAdapter(list[_Customer])
_PRODUCTS: TypeAdapter[list[_Product]] = TypeAdapter(list[_Product])


def _customer_option(customer: _Customer) -> Option:
    label = f"{customer.fullname} ({customer.shortcode})" if customer.shortcode else customer.fullname
    names = (customer.fullname, customer.shortcode) if customer.shortcode else (customer.fullname,)
    return Option(customer.customerId, label, aliases=names)


class CustomerIdWidget(Widget):
    """``CustomerId``: a customer of core's ``customers`` query, shown by name and shortcode."""

    id = "customerId"

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("type") == "string" and field.get("format") == "customerId"

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        data = await ctx.graphql(CUSTOMERS_QUERY)
        return [_customer_option(customer) for customer in _CUSTOMERS.validate_python(data["customers"]["page"])]


class ProductIdWidget(Widget):
    """``ProductId`` / ``product_id([...])``: a product of core's catalogue, restricted to its ``productIds``."""

    id = "productId"

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("format") == "productId"

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        allowed = {str(product_id) for product_id in field_hint(field, "productIds") or ()}
        products = _PRODUCTS.validate_python(await ctx.call_tool(LIST_PRODUCTS_TOOL, {}))
        return [
            Option(product.product_id, product.name)
            for product in products
            if not allowed or product.product_id in allowed
        ]


BUILTIN_WIDGETS: tuple[FieldWidget, ...] = (CustomerIdWidget(), ProductIdWidget())

__all__ = ["BUILTIN_WIDGETS", "CUSTOMERS_QUERY", "CustomerIdWidget", "ProductIdWidget"]
```

Note: the `restricted` test expects core's order (`P1, P2`), not the hint's order — options keep the catalogue order.

- [ ] **Step 5: Re-export** — in `widgets/__init__.py` add
`from orchestrator_agent.form_fill.widgets.core import BUILTIN_WIDGETS, CustomerIdWidget, ProductIdWidget`
and the three names to `__all__`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form_widgets_core.py tests/test_tool_contract.py -q`
Expected: all pass.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/orchestrator_agent tests/test_form_widgets_core.py
uv run ruff format src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/tool_names.py tests/test_form_widgets_core.py
git add src/orchestrator_agent/form_fill/widgets src/orchestrator_agent/tool_names.py tests/test_form_widgets_core.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Add the built-in customerId and productId widgets"
```

---

### Task 4: `enrich` — a page schema with the widgets' options in it

**Files:**
- Create: `src/orchestrator_agent/form_fill/widgets/enrich.py`
- Modify: `src/orchestrator_agent/form_fill/core_bridge.py` (carry `x-widget`, labels of a long list, `widget_mark`)
- Modify: `src/orchestrator_agent/form_fill/widgets/__init__.py` (re-export)
- Test: `tests/test_form_widgets_enrich.py`

**Interfaces:**
- Consumes: `match_widget`, `widget_target`, `Option`, `WidgetContext`, `MAX_INLINE`, `WIDGET_MARK` (Task 1); `core_bridge.resolve_property`, `page_model`, `choices`, `labels` (existing).
- Produces:
  - `enrich(schema: Mapping[str, Any], widgets: Sequence[FieldWidget], ctx: WidgetContext) -> EnrichedPage`.
  - `EnrichedPage(schema: dict[str, Any], long_lists: dict[str, LongList])` — frozen dataclass; `schema` is what is stored and turned into the page model.
  - `LongList(widget: FieldWidget, field: dict[str, Any], options: tuple[Option, ...], title: str = "", multiple: bool = False)` — frozen dataclass: a long-list field's widget, its resolved property (the items' schema for an array), every option, the field's title and whether it is a list (used by Task 5).
  - Marks written on a property: `{"x-widget": {"id": str}}` (inlined), `{"x-widget": {"id": str, "total": int}}` (long list), `{"x-widget": {"id": str, "later": True}}` (waits).
  - `core_bridge.widget_mark(info: FieldInfo) -> dict[str, Any] | None` — the mark of a page-model field.

Rewriting rules (`enrich` never touches a property no widget matches):

| Widget result | Property after `enrich` |
|---|---|
| ≤ `MAX_INLINE` options | the resolved property + `enum` (values) + `options` (value → label) + mark `{id}`; for an array the `enum`/`options` go into `items` |
| > `MAX_INLINE` | the resolved property + mark `{id, total}`; the options go to `long_lists` |
| `None` | the resolved property + mark `{id, later: true}` |
| raised | unchanged; `logger.warning("Form-fill widget failed", widget=id, field=name, error=str(exc))` |

"The resolved property" is `resolve_property(prop, defs)` (flattens `$ref` and the nullable `anyOf`); its `default` is kept, so an optional field stays optional. `ctx.values` is passed as it is (the skill fills it).

- [ ] **Step 1: Write the failing tests** — create `tests/test_form_widgets_enrich.py`:

```python
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
    assert page.long_lists == {}
    fields = page_model(page.schema).model_fields
    assert choices(fields["customer_id"]) == ("c-0", "c-1", "c-2") and fields["customer_id"].is_required()
    assert labels(fields["customer_id"]) == {"c-0": "Customer 0", "c-1": "Customer 1", "c-2": "Customer 2"}
    assert choices(fields["backup"]) == ("c-0", "c-1", "c-2") and not fields["backup"].is_required()
    assert is_list(fields["customers"]) and choices(fields["customers"]) == ("c-0", "c-1", "c-2")
    assert widget_mark(fields["customer_id"]) == {"id": "customerId"}
    assert widget_mark(fields["note"]) is None and fields["note"].annotation == str | None  # untouched


async def test_a_long_list_is_marked_and_kept_for_resolving():
    page = await enrich(PAGE, [Customers(MANY)], CTX)
    assert set(page.long_lists) == {"customer_id", "backup", "customers"}
    customer = page.long_lists["customer_id"]
    assert customer.options == tuple(MANY) and customer.field["format"] == "customerId"
    assert customer.title == "Customer Id" and customer.multiple is False
    customers = page.long_lists["customers"]
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
    assert ("customer_id" not in page.long_lists) is inlined


async def test_options_that_wait_for_another_value_mark_the_field():
    page = await enrich(PAGE, [Customers(None)], CTX)
    assert page_model(page.schema).model_fields["customer_id"].json_schema_extra["x-widget"] == {  # type: ignore[index]
        "id": "customerId",
        "later": True,
    }


async def test_a_failing_widget_leaves_the_property_as_it_was():
    page = await enrich(PAGE, [Customers(RuntimeError("core down"))], CTX)
    assert page.schema == PAGE and page.long_lists == {}


async def test_the_options_are_fetched_once_per_page_and_the_schema_is_not_mutated():
    widget = Customers(FEW)
    original = {**PAGE, "properties": dict(PAGE["properties"])}
    await enrich(PAGE, [widget], CTX)
    assert widget.calls == 1  # three fields of one widget with the same hints
    assert PAGE == original
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets_enrich.py -q`
Expected: collection error, `ImportError: cannot import name 'widget_mark'` (or the missing `enrich` module).

- [ ] **Step 3: Carry the mark through `core_bridge`** — in `src/orchestrator_agent/form_fill/core_bridge.py`:

Add the constant next to `LABELS`:

```python
WIDGET = "x-widget"  # what a form-fill widget wrote on a field (``widgets.enrich``)
```

In `_field`, after the `labels` line, carry the mark (`enrich` always writes it on the property itself, an array's
too):

```python
    if mark := prop.get(WIDGET):
        extra[WIDGET] = mark
```

Add after `is_accept`:

```python
def widget_mark(info: FieldInfo) -> dict[str, Any] | None:
    """What a form-fill widget wrote on the field: its id, and ``total`` for a long list or ``later`` while it waits."""
    extra = info.json_schema_extra
    mark = extra.get(WIDGET) if isinstance(extra, dict) else None
    return mark if isinstance(mark, dict) else None
```

and add `"WIDGET"` and `"widget_mark"` to `__all__`.

- [ ] **Step 4: Write `src/orchestrator_agent/form_fill/widgets/enrich.py`** (license header first):

```python
"""A page schema with the widgets' options written in, the way core would have sent a static list.

A field a widget knows is rewritten before the page model is built: a short list becomes core's own choice
shape (``enum`` + ``options``), so the stops, the labels and the approval need nothing new; a long list is
marked and kept aside, and what a person types for it is resolved to one of its options before core sees it
(``widgets.resolve``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from orchestrator_agent.form_fill.core_bridge import resolve_property
from orchestrator_agent.form_fill.widgets.base import (
    MAX_INLINE,
    WIDGET_MARK,
    FieldWidget,
    Option,
    WidgetContext,
)
from orchestrator_agent.form_fill.widgets.registry import match_widget, widget_target

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LongList:
    """A field with more options than a question shows: its widget, its (items') property, every option."""

    widget: FieldWidget
    field: dict[str, Any]
    options: tuple[Option, ...]
    title: str = ""
    multiple: bool = False


@dataclass(frozen=True)
class EnrichedPage:
    schema: dict[str, Any]
    long_lists: dict[str, LongList] = field(default_factory=dict)


@dataclass(frozen=True)
class _Property:
    """One property of the page after ``enrich``: as it goes into the schema, and its long list if it is one."""

    name: str
    schema: Any
    long_list: LongList | None = None


def _choice(target: Mapping[str, Any], options: Sequence[Option]) -> dict[str, Any]:
    return {**target, "enum": [o.value for o in options], "options": {str(o.value): o.label for o in options}}


def _with_target(prop: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {**prop, "items": target} if prop.get("type") == "array" else target


class _Fetches:
    """The options each widget has on one page, fetched once per widget and hints."""

    def __init__(self, ctx: WidgetContext) -> None:
        self.ctx = ctx
        self.seen: dict[tuple[str, str], Sequence[Option] | None] = {}

    async def __call__(self, widget: FieldWidget, target: Mapping[str, Any]) -> Sequence[Option] | None:
        hints = {key: target.get(key) for key in ("format", "extraProperties", "uniforms")}
        key = (widget.id, json.dumps(hints, sort_keys=True, default=str))
        if key not in self.seen:
            self.seen[key] = await widget.options(target, self.ctx)
        return self.seen[key]


async def enrich(schema: Mapping[str, Any], widgets: Sequence[FieldWidget], ctx: WidgetContext) -> EnrichedPage:
    """``schema`` with every property a widget matches rewritten; every other property as it came."""
    defs = schema.get("$defs") or {}
    fetch = _Fetches(ctx)
    found = [await _property(name, prop, defs, widgets, fetch) for name, prop in (schema.get("properties") or {}).items()]
    long_lists = {p.name: p.long_list for p in found if p.long_list is not None}
    if not found:
        return EnrichedPage(dict(schema))
    return EnrichedPage({**schema, "properties": {p.name: p.schema for p in found}}, long_lists)


async def _property(
    name: str, prop: Any, defs: Mapping[str, Any], widgets: Sequence[FieldWidget], fetch: _Fetches
) -> _Property:
    """One property after ``enrich``: unchanged unless a widget matches it and could say what its options are."""
    resolved = resolve_property(prop, defs) if isinstance(prop, Mapping) else {}
    widget = match_widget(widgets, resolved) if resolved else None
    if widget is None:
        return _Property(name, prop)
    target = resolve_property(widget_target(resolved), defs)
    try:
        options = await fetch(widget, target)
    except Exception as exc:  # a widget is an add-on: without it the field is asked as before
        logger.warning("Form-fill widget failed", widget=widget.id, field=name, error=str(exc))
        return _Property(name, prop)
    if options is None:
        return _Property(name, {**resolved, WIDGET_MARK: {"id": widget.id, "later": True}})
    if len(options) <= MAX_INLINE:
        return _Property(name, {**_with_target(resolved, _choice(target, options)), WIDGET_MARK: {"id": widget.id}})
    marked = {**_with_target(resolved, target), WIDGET_MARK: {"id": widget.id, "total": len(options)}}
    title = str(resolved.get("title") or name)
    long_list = LongList(widget, target, tuple(options), title=title, multiple=resolved.get("type") == "array")
    return _Property(name, marked, long_list)


__all__ = ["EnrichedPage", "LongList", "enrich"]
```

- [ ] **Step 5: Re-export** — in `widgets/__init__.py` add
`from orchestrator_agent.form_fill.widgets.enrich import EnrichedPage, LongList, enrich` and the names to `__all__`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form_widgets_enrich.py tests/test_form_fill.py -q`
Expected: all pass (the existing page-model tests are unaffected: their schemas carry no `x-widget`).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/orchestrator_agent/form_fill tests/test_form_widgets_enrich.py
uv run ruff format src/orchestrator_agent/form_fill tests/test_form_widgets_enrich.py
git add src/orchestrator_agent/form_fill tests/test_form_widgets_enrich.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Write widget options into a form page's schema"
```

---

### Task 5: Resolving typed words to one option of a long list

**Files:**
- Create: `src/orchestrator_agent/form_fill/widgets/resolve.py`
- Modify: `src/orchestrator_agent/form_fill/interpret.py` (`Chooser`, `choice_schema`, `ModelInterpreter.choose`)
- Modify: `src/orchestrator_agent/form_fill/widgets/__init__.py` (re-export)
- Test: `tests/test_form_widgets_resolve.py`

**Interfaces:**
- Consumes: `LongList` (Task 4); `Option`, `WidgetContext`, `MAX_FULL_READ`, `names_of` (Task 1); `ModelInterpreter`, `INSTRUCTIONS` (existing `interpret.py`).
- Produces:
  - In `interpret.py`: `Chooser` protocol — `async choose(title: str, options: Sequence[tuple[Any, str]], words: str) -> list[Any]` (every value the words fit, in no particular order); `ModelInterpreter.choose` implements it (one run, output type `list[Literal[values]]`).
  - `exact_options(options: Sequence[Option], words: str) -> list[Option]` — options with a name (label, value, alias) equal to the words, case-insensitive.
  - `Resolution` — frozen dataclass: `value: Any = None` (the resolved value, set when `resolved`), `resolved: bool = False`, `candidates: tuple[Option, ...] = ()` (several fit), `unmatched: tuple[str, ...] = ()` (words nothing fit).
  - `async resolve_words(long_list: LongList, words: str, ctx: WidgetContext, chooser: Chooser | None) -> Resolution` — one item's words.
  - `async resolve_answer(long_list: LongList, answer: Any, ctx: WidgetContext, chooser: Chooser | None) -> Resolution` — a field's answer: a value already among the options is taken as it is; a list field resolves each item (a single typed string is split on `,`); the answer is resolved only when every item is.

Order inside `resolve_words` (exact before candidates before the model, per the spec):
1. `exact_options(long_list.options, words)`: one → resolved, no model call; several → `candidates` (shared name).
2. Candidates: with a chooser, `long_list.options` in full when there are at most `MAX_FULL_READ` (abbreviations still resolve); otherwise — and always without a chooser — the narrowed ones: `narrow_options(long_list.options, words)` for a list of at most `MAX_FULL_READ`, else `await long_list.widget.options(long_list.field, ctx, search=words) or ()`. None left → `unmatched`.
3. No chooser → the narrowed candidates are offered when there are at most `MAX_INLINE`, else `unmatched`.
4. Chooser → `fits = [v for v in await chooser.choose(...) if v in candidate values]` (a value outside the candidates is ignored): one → resolved; several → `candidates` of those; none → `unmatched`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_form_widgets_resolve.py`:

```python
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
from orchestrator_agent.form_fill.widgets.resolve import Resolution, exact_options, resolve_answer, resolve_words

from .test_form_widgets import no_graphql, no_tool

CUSTOMERS = (
    Option("c-ut", "Universiteit Twente (UT)", aliases=("Universiteit Twente", "UT")),
    Option("c-uu", "Universiteit Utrecht (UU)", aliases=("Universiteit Utrecht", "UU")),
    Option("c-ta", "Testaccount (TA)", aliases=("Testaccount", "TA")),
    Option("c-x1", "SURF (SURF)", aliases=("SURF",)),
    Option("c-x2", "SURF (SURF2)", aliases=("SURF", "SURF2")),
    *(Option(f"c-{n}", f"Customer {n:02d}") for n in range(20)),
)
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
        self.picks, self.calls = picks, []

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


async def test_a_list_longer_than_the_full_read_is_searched_first():
    many = tuple(Option(f"c-{n}", f"Customer {n:03d}") for n in range(250))
    widget = Searchable(many)
    chooser = FakeChooser(["c-7"])
    resolution = await resolve_words(long_list(many, widget=widget), "customer 7", CTX, chooser)
    assert resolution == Resolution(value="c-7", resolved=True)
    assert widget.searches == ["customer 7"] and len(chooser.calls[0][1]) == 50  # the widget's search, capped


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
        pytest.param(
            ["c-ut", "Testaccount"], True, Resolution(value=["c-ut", "c-ta"], resolved=True), id="list-items"
        ),
        pytest.param(
            "UT, testaccount", True, Resolution(value=["c-ut", "c-ta"], resolved=True), id="list-typed-commas"
        ),
        pytest.param("UT, nobody", True, Resolution(unmatched=("nobody",)), id="list-one-item-unmatched"),
    ],
)
async def test_a_fields_answer(answer, multiple, expected):
    assert await resolve_answer(long_list(multiple=multiple), answer, CTX, FakeChooser([])) == expected


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets_resolve.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'orchestrator_agent.form_fill.widgets.resolve'`.

- [ ] **Step 3: Add `choose` to the interpreter** — in `src/orchestrator_agent/form_fill/interpret.py`:

Add after the `Interpreter` protocol:

```python
class Chooser(Protocol):
    """Which of a field's options a person's words fit: every value they fit, so the caller sees when several do."""

    async def choose(self, title: str, options: Sequence[tuple[Any, str]], words: str) -> list[Any]: ...


CHOOSE_INSTRUCTIONS = (
    "A person answered a form field in their own words. The field's options are given as values with how they are "
    "shown to people. Give every option value the words fit — a name, a short name, a description or a part of "
    "one — so more than one when the words do not tell the options apart, and none when they fit none. Never give "
    "a value that is not one of the options."
)
```

(import `Sequence` from `collections.abc` and `Literal` from `typing`). Add a method to `ModelInterpreter`:

```python
    async def choose(self, title: str, options: Sequence[tuple[Any, str]], words: str) -> list[Any]:
        values = tuple(value for value, _ in options)
        if not values:
            return []
        reading = create_model("Choice", values=(list[Literal.__getitem__(values)], Field(default_factory=list)))  # type: ignore[misc]
        agent: Agent[None, Any] = Agent(self.model, output_type=reading, instructions=CHOOSE_INSTRUCTIONS)
        listed = "\n".join(f"- {json.dumps(value)}: {label}" for value, label in options)
        prompt = f"Field: {title}\nOptions (value: shown as):\n{listed}\nAnswer: {words}"
        chosen = list((await agent.run(prompt)).output.values)
        logger.info("Form-fill option chosen", field=title, words=words, chosen=chosen)
        return chosen
```

Add `"CHOOSE_INSTRUCTIONS"` and `"Chooser"` to `__all__`.

- [ ] **Step 4: Write `src/orchestrator_agent/form_fill/widgets/resolve.py`** (license header first):

```python
"""What a person typed for a long-list field, resolved to one of its options before core sees it.

Core accepts anything for some of these fields (``CustomerId`` is a plain string), so words never travel as a
value: an exact name is its option, otherwise the interpreter reads the words over the candidates and only an
answer that fits exactly one of them is taken. Several fits are offered to pick from; none is asked again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator_agent.form_fill.interpret import Chooser
from orchestrator_agent.form_fill.widgets.base import (
    MAX_FULL_READ,
    MAX_INLINE,
    Option,
    WidgetContext,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.enrich import LongList


@dataclass(frozen=True)
class Resolution:
    """The outcome for one answer: a value, the options it may mean, or the words nothing fits."""

    value: Any = None
    resolved: bool = False
    candidates: tuple[Option, ...] = ()
    unmatched: tuple[str, ...] = ()


def exact_options(options: Sequence[Option], words: str) -> list[Option]:
    """The options one of whose names (label, value, alias) are the words, but for case and outer spaces."""
    wanted = words.strip().casefold()
    return [option for option in options if any(name.casefold() == wanted for name in names_of(option))]


async def _candidates(long_list: LongList, words: str, ctx: WidgetContext, *, full: bool) -> tuple[Option, ...]:
    """What the words are read against: a short enough list in ``full``, else the options a search finds."""
    if len(long_list.options) <= MAX_FULL_READ:
        return long_list.options if full else tuple(narrow_options(long_list.options, words))
    return tuple(await long_list.widget.options(long_list.field, ctx, search=words) or ())


async def resolve_words(long_list: LongList, words: str, ctx: WidgetContext, chooser: Chooser | None) -> Resolution:
    """One item's words: an exact name, else what the chooser says the words fit among the candidates."""
    exact = exact_options(long_list.options, words)
    if len(exact) == 1:
        return Resolution(value=exact[0].value, resolved=True)
    if exact:
        return Resolution(candidates=tuple(exact))
    candidates = await _candidates(long_list, words, ctx, full=chooser is not None)
    if not candidates:
        return Resolution(unmatched=(words,))
    if chooser is None:
        return Resolution(candidates=candidates) if len(candidates) <= MAX_INLINE else Resolution(unmatched=(words,))
    by_value = {option.value: option for option in candidates}
    picked = await chooser.choose(long_list.title, [(o.value, o.label) for o in candidates], words)
    fits = list(dict.fromkeys(value for value in picked if value in by_value))
    match fits:
        case [value]:
            return Resolution(value=value, resolved=True)
        case []:
            return Resolution(unmatched=(words,))
        case _:
            return Resolution(candidates=tuple(by_value[value] for value in fits))


def _items(answer: Any, multiple: bool) -> list[Any]:
    match answer:
        case list():
            return answer
        case str() if multiple:
            return [part.strip() for part in answer.split(",") if part.strip()]
        case _:
            return [answer]


async def resolve_answer(long_list: LongList, answer: Any, ctx: WidgetContext, chooser: Chooser | None) -> Resolution:
    """A field's answer: a value among the options stands; words are resolved; a list only when every item is."""
    values = {option.value for option in long_list.options}
    outcomes = [
        Resolution(value=item, resolved=True) if item in values else await resolve_words(long_list, str(item), ctx, chooser)
        for item in _items(answer, long_list.multiple)
    ]
    if all(outcome.resolved for outcome in outcomes):
        resolved = [outcome.value for outcome in outcomes]
        return Resolution(value=resolved if long_list.multiple else resolved[0], resolved=True)
    unmatched = tuple(words for outcome in outcomes for words in outcome.unmatched)
    candidates = tuple(option for outcome in outcomes for option in outcome.candidates)
    return Resolution(candidates=candidates, unmatched=unmatched)


__all__ = ["Resolution", "exact_options", "resolve_answer", "resolve_words"]
```

An empty answer list never reaches `resolve_answer` (the skill only resolves non-empty answers), so `resolved[0]` is safe.

Also add to `resolve.py` (and `__all__`) how a resolved value is shown, used by the skill for its labels:

```python
def shown_as(long_list: LongList, value: Any) -> Any:
    """How a resolved value is shown: its option's label (each item's, for a list)."""
    labels = {option.value: option.label for option in long_list.options}
    return [labels.get(item, item) for item in value] if isinstance(value, list) else labels.get(value, value)
```

with this test in `tests/test_form_widgets_resolve.py` (import `shown_as`):

```python
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
```

- [ ] **Step 5: Re-export** — in `widgets/__init__.py` add
`from orchestrator_agent.form_fill.widgets.resolve import Resolution, exact_options, resolve_answer, resolve_words`
and the names to `__all__`. Note `widgets/__init__.py` must import `resolve` after `enrich` (it imports `LongList`).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form_widgets_resolve.py tests/test_form_interpret.py -q`
Expected: all pass.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/orchestrator_agent/form_fill tests/test_form_widgets_resolve.py
uv run ruff format src/orchestrator_agent/form_fill tests/test_form_widgets_resolve.py
git add src/orchestrator_agent/form_fill tests/test_form_widgets_resolve.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Resolve typed words to one option of a long-list widget field"
```

---

### Task 6: The skill walks widget fields

**Files:**
- Modify: `src/orchestrator_agent/state.py` (`AskField.hint`, `FormFillSession.resolved`)
- Modify: `src/orchestrator_agent/form_fill/skill.py`
- Test: `tests/test_form_widgets_skill.py`

**Interfaces:**
- Consumes: `enrich`, `EnrichedPage`, `LongList` (Task 4); `resolve_answer`, `Resolution`, `shown_as` (Task 5); `FieldWidget`, `GraphQL`, `WidgetContext`, `MAX_INLINE` (Task 1); `Chooser` (Task 5); `core_bridge.widget_mark` (Task 4).
- Produces:
  - `AskField.hint: str = ""` — one line a transport shows with the question (Task 7 renders it).
  - `FormFillSession.resolved: dict[str, dict[str, Any]]` — field → `{"value": ..., "label": ...}` of what its typed words resolved to.
  - `FormFillSkill(interpret=None, widgets=(), graphql=None, choose=None)` — new keyword fields `widgets: Sequence[FieldWidget]`, `graphql: GraphQL | None`, `choose: Chooser | None`.
  - Module constants in `skill.py`: `LONG_LIST_HINT = "Type a name or part of it — {total} options."`, `SEVERAL_FIT = "More than one option fits what was typed: pick one, or type more of the name."`, `TOO_MANY_FIT = "Many options fit what was typed: type more of the name."`, `NOTHING_FIT = "Nothing matched {words}: type another name."`.
  - `offered(reply: Reply, unresolved: Mapping[str, Resolution]) -> Reply`.

Behaviour (the walk, per page core returns):
1. `enrich` the schema with `WidgetContext(call_tool, self.graphql or _no_graphql, values={**values of the pages walked so far, **session.values})`; store and model the **enriched** schema.
2. Resolve every answered long-list field (`resolve_answer` with `self.choose`): resolved → `session.values[name] = value` and `session.resolved[name] = {"value", "label": shown_as(...)}`; not resolved → the words are dropped from `session.values` (and `session.resolved`), so they never reach core.
3. Collect the page's values as before (`_see_page`); if any field did not resolve, stop at this page **without submitting it** and offer, per field: the candidates as chips when 1–`MAX_INLINE` fit (`SEVERAL_FIT`), `TOO_MANY_FIT` when more fit, `NOTHING_FIT` otherwise.
4. `question()` gives a long-list field `LONG_LIST_HINT`; `questions()` leaves out fields marked `later` (and core's errors on them).
5. `_reinterpret` never reads a widget field (core rejecting a picked option is the person's to fix).
6. `_labels` adds the label of a resolved value while the field still holds that value.

- [ ] **Step 1: Write the failing tests** — create `tests/test_form_widgets_skill.py`:

```python
"""The skill walking a page with widget fields: options as chips, typed words resolved, words never sent to core."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill.pending import pending_of
from orchestrator_agent.form_fill.skill import NOTHING_FIT, SEVERAL_FIT, FormFillSkill
from orchestrator_agent.form_fill.widgets import Option
from orchestrator_agent.state import FormFillSession, SearchState

from .test_form_fill import FakeCore, open_form, rejection, turn
from .test_form_widgets import FormatWidget

PAGE = {
    "properties": {
        "customer_id": {"format": "customerId", "title": "Customer", "type": "string"},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Note"},
    },
    "required": ["customer_id"],
    "title": "Customer",
    "type": "object",
}
UT = Option("c-ut", "Universiteit Twente (UT)", aliases=("Universiteit Twente", "UT"))
UU = Option("c-uu", "Universiteit Utrecht (UU)", aliases=("Universiteit Utrecht", "UU"))
TA = Option("c-ta", "Testaccount (TA)", aliases=("Testaccount", "TA"))
MANY = (TA, UT, UU, *(Option(f"c-{n}", f"Customer {n:02d}") for n in range(20)))
KEY = "widget_demo"


class WidgetCore:
    """Core's form tools for a one-page task asking a customer; like core's ``CustomerId`` it takes any string."""

    def __init__(self, refuse: str | None = None) -> None:
        self.refuse = refuse  # a customer id core's validator rejects
        self.submitted: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        if name == "list_workflows":
            return [{**FakeCore.WORKFLOWS[0], "name": KEY}]
        if name == "create_workflow":
            self.created.append(args)
            return {"id": FakeCore.PROCESS_ID}
        assert name == "get_workflow_form"
        inputs = args["page_inputs"]
        self.submitted.extend(inputs)
        if not inputs:
            return {"page": 0, "complete": False, "schema": PAGE}
        FakeCore.require(PAGE, inputs[0])
        if inputs[0]["customer_id"] == self.refuse:
            raise ModelRetry(rejection({"customer_id": "Customer not allowed"}))
        return {"page": 1, "complete": True, "schema": None}


class Chooser:
    def __init__(self, picks: list[Any]) -> None:
        self.picks, self.calls = picks, []

    async def choose(self, title, options, words):
        self.calls.append(words)
        return self.picks


class Refuses:
    async def choose(self, title, options, words):
        raise AssertionError("no model call expected")

    async def answers(self, form, words):
        raise AssertionError("no interpretation expected")


def make(options: Sequence[Option] | None = MANY, choose=None, interpret=None) -> FormFillSkill:
    return FormFillSkill(widgets=[FormatWidget("customerId", "customerId", options)], choose=choose, interpret=interpret)


async def test_a_long_list_is_asked_as_typed_text_with_its_size():
    reply = await open_form(make(), WidgetCore(), SearchState(), KEY)
    customer = reply.question("customer_id")
    assert customer.choices == () and customer.hint == "Type a name or part of it — 23 options."


async def test_an_exact_name_is_its_option_without_a_model():
    core, state = WidgetCore(), SearchState()
    reply = await open_form(make(choose=Refuses()), core, state, KEY, {"customer_id": "TESTACCOUNT"})
    assert reply.status == "confirming" and reply.values["customer_id"] == "c-ta"
    assert reply.labels == {"customer_id": "Testaccount (TA)"}
    assert reply.approval.args["labels"] == {"customer_id": "Testaccount (TA)"}
    assert "TESTACCOUNT" not in str(core.submitted)


async def test_words_that_fit_one_option_are_that_option():
    core, chooser = WidgetCore(), Chooser(["c-ut"])
    reply = await open_form(make(choose=chooser), core, SearchState(), KEY, {"customer_id": "twente"})
    assert reply.status == "confirming" and reply.values["customer_id"] == "c-ut" and chooser.calls == ["twente"]


@pytest.mark.parametrize(
    "picks,choices,hint",
    [
        pytest.param(["c-ut", "c-uu"], (UT.label, UU.label), SEVERAL_FIT, id="several-fit"),
        pytest.param([], (), NOTHING_FIT.format(words="'uni'"), id="nothing-fits"),
    ],
)
async def test_unresolved_words_are_never_submitted(picks, choices, hint):
    core, state = WidgetCore(), SearchState()
    reply = await open_form(make(choose=Chooser(picks)), core, state, KEY, {"customer_id": "uni"})
    customer = reply.question("customer_id")
    assert reply.status == "gathering" and customer.choices == choices and customer.hint == hint
    assert "uni" not in str(core.submitted) and "customer_id" not in state.form_fill.values


async def test_a_picked_candidate_continues_and_is_not_read_again():
    core, state, chooser = WidgetCore(), SearchState(), Chooser(["c-ut", "c-uu"])
    skill = make(choose=chooser)
    await open_form(skill, core, state, KEY, {"customer_id": "uni"})
    reply = await turn(skill, core, state, {"customer_id": "c-uu"})  # the chip's value, as the transport maps it
    assert reply.status == "confirming" and reply.labels == {"customer_id": "Universiteit Utrecht (UU)"}
    await turn(skill, core, state, {"note": "hello"})  # another walk: the picked value stands
    assert chooser.calls == ["uni"]


async def test_a_short_list_is_chips_and_a_single_option_is_taken():
    reply = await open_form(make(options=(UT, UU)), WidgetCore(), SearchState(), KEY)
    customer = reply.question("customer_id")
    assert customer.choices == (UT.label, UU.label) and customer.values == ("c-ut", "c-uu") and customer.hint == ""
    only = await open_form(make(options=(TA,)), WidgetCore(), SearchState(), KEY)  # plain core: one default customer
    assert only.status == "confirming" and only.values["customer_id"] == "c-ta"


async def test_a_field_waiting_for_another_value_is_not_asked():
    reply = await open_form(make(options=None), WidgetCore(), SearchState(), KEY)
    assert "customer_id" not in reply.asked


async def test_core_rejecting_a_picked_option_is_not_reinterpreted():
    skill = make(choose=Refuses(), interpret=Refuses())
    reply = await open_form(skill, WidgetCore(refuse="c-ut"), SearchState(), KEY, {"customer_id": "UT"})
    assert reply.question("customer_id").problem == "Customer not allowed"


def test_a_session_persisted_before_widgets_still_loads():
    question = {"name": "a", "question": "A?", "choices": [], "values": [], "multiple": False, "required": True}
    old = {"workflow_key": KEY, "status": "gathering", "pending": {"id": "p", "kind": "ask", "questions": [question]}}
    session = FormFillSession.model_validate(old)
    pending = pending_of(session)
    assert session.resolved == {} and pending is not None and pending.questions[0].hint == ""
```

`FormatWidget` (from `tests/test_form_widgets.py`, Task 1) returns its options for any property of its format.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets_skill.py -q`
Expected: collection error, `ImportError: cannot import name 'NOTHING_FIT' from 'orchestrator_agent.form_fill.skill'`.

- [ ] **Step 3: Extend the state** — in `src/orchestrator_agent/state.py`:

In `AskField`, after `problem`:

```python
    hint: str = ""  # one line shown with the question: how many options a long list has, what typed words matched
```

In `FormFillSession`, after `interpreted`:

```python
    # long-list widget field -> what the person's words resolved to: {"value": <option value>, "label": <its label>}
    resolved: dict[str, dict[str, Any]] = Field(default_factory=dict)
```

- [ ] **Step 4: Change the skill** — in `src/orchestrator_agent/form_fill/skill.py`:

Imports: add `from dataclasses import replace`, extend the `core_bridge` import with `widget_mark`, change the interpret import to
`from orchestrator_agent.form_fill.interpret import Chooser, Interpreter`, and add:

```python
from orchestrator_agent.form_fill.widgets import MAX_INLINE, FieldWidget, GraphQL, WidgetContext
from orchestrator_agent.form_fill.widgets.enrich import LongList, enrich
from orchestrator_agent.form_fill.widgets.resolve import Resolution, resolve_answer, shown_as
```

Constants, after `_MISSING`:

```python
# What a stop says about a long-list widget field (``AskField.hint``).
LONG_LIST_HINT = "Type a name or part of it — {total} options."
SEVERAL_FIT = "More than one option fits what was typed: pick one, or type more of the name."
TOO_MANY_FIT = "Many options fit what was typed: type more of the name."
NOTHING_FIT = "Nothing matched {words}: type another name."


async def _no_graphql(query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raise RuntimeError("no GraphQL endpoint of core is configured for the form-fill widgets")
```

In `questions()`, leave out the fields that wait for another value — replace the first line `fields = model.model_fields` with:

```python
    waiting = {name for name, info in model.model_fields.items() if (widget_mark(info) or {}).get("later")}
    fields = {name: info for name, info in model.model_fields.items() if name not in waiting}
    errors = [error for error in errors if not (error["loc"] and str(error["loc"][0]) in waiting)]
```

In `question()`, give a long list its hint — before `return AskField(`:

```python
    mark = widget_mark(info) or {}
    hint = LONG_LIST_HINT.format(total=mark["total"]) if "total" in mark else ""
```

and pass `hint=hint,` to `AskField(...)`.

Add after `question()`:

```python
def offered(reply: Reply, unresolved: Mapping[str, Resolution]) -> Reply:
    """The stop, with what resolving said about each field whose words it could not resolve."""
    if not reply.ask or not unresolved:
        return reply
    return replace(reply, ask=[_offer(field, unresolved.get(field.name)) for field in reply.ask])


def _offer(field: AskField, outcome: Resolution | None) -> AskField:
    """A field asked again: the options that fit as chips, or why nothing is offered."""
    match outcome:
        case None:
            return field
        case Resolution(candidates=candidates) if 0 < len(candidates) <= MAX_INLINE:
            labels, values = tuple(o.label for o in candidates), tuple(o.value for o in candidates)
            return replace(field, choices=labels, values=values, hint=SEVERAL_FIT)
        case Resolution(candidates=candidates) if candidates:
            return replace(field, hint=TOO_MANY_FIT)
        case _:
            return replace(field, hint=NOTHING_FIT.format(words=", ".join(f"'{w}'" for w in outcome.unmatched)))


def _record(session: FormFillSession, name: str, long_list: LongList, outcome: Resolution) -> None:
    """Keep what the words resolved to in place of the words; drop words that did not resolve (core never sees them)."""
    if outcome.resolved:
        session.values[name] = outcome.value
        session.resolved[name] = {"value": outcome.value, "label": shown_as(long_list, outcome.value)}
    else:
        session.values.pop(name, None)
        session.resolved.pop(name, None)
```

`FormFillSkill` fields — after `interpret`:

```python
    widgets: Sequence[FieldWidget] = ()  # the formats whose options core leaves to the frontend (``form_fill.widgets``)
    graphql: GraphQL | None = None  # core's GraphQL API, for the widgets that read it
    choose: Chooser | None = None  # typed words for a long-list widget field -> the options they fit
```

In `_walk`, replace the three lines

```python
            schema = page.schema_ or {}
            session.pages.append(schema)
            model = page_model(schema)
            values = self._see_page(session, model, pages)
```

with

```python
            ctx = self._widget_context(session, call_tool, pages)
            enriched = await enrich(page.schema_ or {}, self.widgets, ctx)
            session.pages.append(enriched.schema)
            model = page_model(enriched.schema)
            unresolved = await self._resolve(session, enriched.long_lists, ctx)
            values = self._see_page(session, model, pages)
            if unresolved:  # words that are no option: the page is asked again, never submitted with them
                return offered(self._page_stop(session, [*pages, values]), unresolved)
```

Add the two helpers to `FormFillSkill` (after `_walk`):

```python
    def _widget_context(self, session: FormFillSession, call_tool: CallTool, pages: list[dict[str, Any]]) -> WidgetContext:
        """What a widget may use on this page: core's APIs, and every value known so far."""
        known = {name: value for page in pages for name, value in page.items()}
        return WidgetContext(call_tool=call_tool, graphql=self.graphql or _no_graphql, values={**known, **session.values})

    async def _resolve(
        self, session: FormFillSession, long_lists: Mapping[str, LongList], ctx: WidgetContext
    ) -> dict[str, Resolution]:
        """Each answered long-list field of the page resolved to its options in place; those that were not, by name.

        A resolved value replaces the words, so the next walk finds it among the options and reads nothing again.
        """
        answered = {name: ll for name, ll in long_lists.items() if session.values.get(name) not in (None, "", [])}
        outcomes = {
            name: await resolve_answer(ll, session.values[name], ctx, self.choose) for name, ll in answered.items()
        }
        for name, outcome in outcomes.items():
            _record(session, name, long_lists[name], outcome)
        return {name: outcome for name, outcome in outcomes.items() if not outcome.resolved}
```

In `_reinterpret`, never read a widget field — change the guard line

```python
            if name not in form.model_fields or name in done or raw is None:
```

to

```python
            if name not in form.model_fields or widget_mark(form.model_fields[name]) or name in done or raw is None:
```

In `_labels`, add the labels of resolved values — replace its body with:

```python
        fields = form_model(session.pages).model_fields
        shown = {name: label_of(fields[name], value) for name, value in values.items() if name in fields}
        resolved = {
            name: kept["label"]
            for name, kept in session.resolved.items()
            if name in values and values[name] == kept["value"]
        }
        return {**resolved, **{name: label for name, label in shown.items() if label is not None}}
```

Add `"LONG_LIST_HINT"`, `"NOTHING_FIT"`, `"SEVERAL_FIT"`, `"TOO_MANY_FIT"`, `"offered"` to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form_widgets_skill.py tests/test_form_fill.py tests/test_form_interpret.py tests/test_form_capability.py -q`
Expected: all pass (the existing skill tests use no widgets, so their schemas pass through `enrich` unchanged).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/orchestrator_agent tests/test_form_widgets_skill.py
uv run ruff format src/orchestrator_agent/form_fill src/orchestrator_agent/state.py tests/test_form_widgets_skill.py
git add src/orchestrator_agent/form_fill src/orchestrator_agent/state.py tests/test_form_widgets_skill.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Walk widget fields in the form-fill skill"
```

---

### Task 7: Hints in both transports, and the skill built with its widgets

**Files:**
- Modify: `src/orchestrator_agent/adapters/chat/librechat.py` (`_question`: hint into the description)
- Modify: `src/orchestrator_agent/adapters/a2a/kagent.py` (`ask_request`: hint after the question)
- Modify: `src/orchestrator_agent/form_fill/__init__.py` (`build_form_fill_skill`)
- Modify: `src/orchestrator_agent/app.py` (log the widgets)
- Test: `tests/test_form_widgets_skill.py` (append)

**Interfaces:**
- Consumes: `AskField.hint` (Task 6); `BUILTIN_WIDGETS` (Task 3); `build_widgets`, `load_extender` (Task 1); `CoreGraphQL`, `graphql_url` (Task 2); `agent_settings.FORM_WIDGET_EXTENDER`, `WFO_CORE_GRAPHQL_URL`, `WFO_CORE_MCP_URL`.
- Produces: `build_form_fill_skill(model=None, *, widgets: Sequence[FieldWidget] | None = None, graphql: GraphQL | None = None) -> FormFillSkill` — `widgets=None` means the built-ins as `FORM_WIDGET_EXTENDER` rearranges them; `graphql=None` means `CoreGraphQL(graphql_url(WFO_CORE_MCP_URL, WFO_CORE_GRAPHQL_URL))`; `ModelInterpreter(model)` is both `interpret` and `choose`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_form_widgets_skill.py`:

```python
from orchestrator_agent.adapters.a2a.kagent import ask_request
from orchestrator_agent.adapters.chat.librechat import ask_card
from orchestrator_agent.adapters.chat.librechat import pause as librechat_pause
from orchestrator_agent.form_fill import build_form_fill_skill
from orchestrator_agent.form_fill.interpret import ModelInterpreter
from orchestrator_agent.form_fill.widgets import CoreGraphQL
from orchestrator_agent.state import AskField, Reply

HINTED = AskField(
    name="customer_id", question="Customer (`customer_id`, required)", title="Customer", hint="Type a name — 23 options."
)


def test_librechat_shows_the_hint_in_the_description():
    session = FormFillSession(workflow_key=KEY, pages=[{"title": "Customer"}])
    pending = librechat_pause(Reply("{}", ask=[HINTED]), session)
    (question,) = ask_card(pending, session, 0)["questions"]
    assert question["description"] == "Type a name — 23 options."


def test_kagent_shows_the_hint_after_the_question():
    payload, _ = ask_request("req-1", [HINTED])
    assert payload.questions[0].question == "Customer (`customer_id`, required) — Type a name — 23 options."


def test_the_skill_is_built_with_the_builtins_as_the_extender_arranges_them(monkeypatch):
    own = FormatWidget("surf-customer", "customerId")
    monkeypatch.setattr("orchestrator_agent.form_fill.agent_settings.FORM_WIDGET_EXTENDER", "fake_widgets:extend")
    monkeypatch.setattr("orchestrator_agent.form_fill.load_extender", lambda path: lambda widgets: [own, *widgets])
    skill = build_form_fill_skill("test")
    assert [widget.id for widget in skill.widgets] == ["surf-customer", "customerId", "productId"]
    assert isinstance(skill.interpret, ModelInterpreter) and skill.choose is skill.interpret
    assert isinstance(skill.graphql, CoreGraphQL) and skill.graphql.url.endswith("/api/graphql")


def test_without_a_model_nothing_reads_typed_words():
    skill = build_form_fill_skill()
    assert skill.interpret is None and skill.choose is None and [w.id for w in skill.widgets] == ["customerId", "productId"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form_widgets_skill.py -q -k "hint or built or reads"`
Expected: FAIL — `KeyError: 'description'` (LibreChat), the question without the hint (kagent), `AttributeError` on `agent_settings` (build).

- [ ] **Step 3: LibreChat** — in `librechat.py` `_question`, change

```python
    described = [field.problem] if field.problem else []
```

to

```python
    described = [text for text in (field.problem, field.hint) if text]
```

- [ ] **Step 4: kagent** — in `kagent.py` `ask_request`, change the questions line to

```python
    questions = [
        HITLQuestion(question=_worded(f), choices=list(f.choices), multiple=f.multiple) for f in ask
    ]
```

and add above `ask_request`:

```python
def _worded(field: AskField) -> str:
    """The question as kagent shows it: its card has no description, so a hint follows the question."""
    return f"{field.question} — {field.hint}" if field.hint else field.question
```

- [ ] **Step 5: Build the skill with its widgets** — replace `build_form_fill_skill` in `src/orchestrator_agent/form_fill/__init__.py`:

```python
from collections.abc import Sequence

from orchestrator_agent.form_fill.widgets import (
    BUILTIN_WIDGETS,
    CoreGraphQL,
    FieldWidget,
    GraphQL,
    build_widgets,
    graphql_url,
    load_extender,
)
from orchestrator_agent.settings import agent_settings


def build_form_fill_skill(
    model: Model | str | None = None,
    *,
    widgets: Sequence[FieldWidget] | None = None,
    graphql: GraphQL | None = None,
) -> FormFillSkill:
    """The skill as configured, with ``model`` reading what a person typed where a value was expected.

    ``model`` interprets typed answers core rejected and chooses the option typed words mean for a long-list
    widget field. The widgets are the built-ins as ``FORM_WIDGET_EXTENDER`` arranges them (a bad extender
    fails here, at startup); they read core's GraphQL API beside its MCP endpoint unless configured otherwise.
    """
    reader = ModelInterpreter(model) if model is not None else None
    return FormFillSkill(
        interpret=reader,
        widgets=build_widgets(BUILTIN_WIDGETS, load_extender(agent_settings.FORM_WIDGET_EXTENDER))
        if widgets is None
        else list(widgets),
        graphql=graphql
        or CoreGraphQL(graphql_url(agent_settings.WFO_CORE_MCP_URL, agent_settings.WFO_CORE_GRAPHQL_URL)),
        choose=reader,
    )
```

- [ ] **Step 6: Log the widgets** — in `src/orchestrator_agent/app.py`, build the skill once and log its widget ids:

```python
    model = agent_settings.create_model()
    form_fill = build_form_fill_skill(model)
    a2a = A2AAdapter(build_agent(model, form_fill=form_fill), url=a2a_url)
```

and add `form_fill_widgets=[widget.id for widget in form_fill.widgets],` to the `logger.info("Agent adapters started", ...)` call.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (`test_kagent_hitl` / `test_librechat_hitl` fixtures carry no hint, so their payloads are unchanged).

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check . && uv run ruff format src tests
git add src tests/test_form_widgets_skill.py
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Show widget hints in both transports and build the skill with its widgets"
```

---

### Task 8: Integration stack — a real orchestrator-core with customers, products and a widget task

**Files:**
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/core/wsgi.py`, `tests/integration/core/main.py`, `tests/integration/core/demo.py`, `tests/integration/core/seed.py`, `tests/integration/core/entrypoint.sh`
- Create: `tests/integration/docker-compose.yml`
- Modify: `pyproject.toml` (register the `integration` marker)
- Test: none yet (Task 9 adds the tests); this task's deliverable is a stack that answers.

**Interfaces:**
- Produces (Task 9 relies on these, verbatim):
  - Core at `http://localhost:8087` (MCP `http://localhost:8087/mcp`, GraphQL `http://localhost:8087/api/graphql`), auth off.
  - 30 customers from GraphQL `customers`: `customerId = f"cust-{n:02d}"`, `fullname = f"Customer {n:02d}"`, `shortcode = f"C{n:02d}"` for `n` in 0–29, except `n == 7`: `fullname = "Testaccount"`, `shortcode = "TA"`.
  - Products `P1 = "d8a3f9b0-1111-4000-8000-000000000001"` ("Widget Port 10G"), `P2 = "…002"` ("Widget Port 1G"), `P3 = "…003"` ("Widget Node", outside the task's `productIds`).
  - Task `widget_demo` (`is_task = true`) with one page: `customer_id: CustomerId`, `product_id: product_id([P1, P2])`, `customers: list[CustomerId] = []`, `note: str = ""`; its one step returns `{"echoed": customer_id}`.

- [ ] **Step 1: Register the marker** — in `pyproject.toml` `[tool.pytest.ini_options]` add:

```toml
markers = ["integration: needs the orchestrator-core stack of tests/integration (WFO_INTEGRATION_CORE_URL)"]
```

- [ ] **Step 2: The core app's domain** — create `tests/integration/core/demo.py`:

```python
"""What the widget integration tests need of a core: thirty customers over GraphQL and a task asking for them."""

from uuid import UUID

import strawberry
from oauth2_lib.strawberry import authenticated_field
from orchestrator.core.forms import FormPage
from orchestrator.core.forms.validators import CustomerId
from orchestrator.core.forms.validators.product_id import product_id
from orchestrator.core.graphql import Query
from orchestrator.core.graphql.pagination import Connection
from orchestrator.core.graphql.schemas.customer import CustomerType
from orchestrator.core.graphql.types import GraphqlFilter, GraphqlSort, OrchestratorInfo
from orchestrator.core.graphql.utils.to_graphql_result_page import to_graphql_result_page
from orchestrator.core.targets import Target
from orchestrator.core.workflow import StepList, done, init, step, workflow
from orchestrator.core.workflows import LazyWorkflowInstance

P1 = UUID("d8a3f9b0-1111-4000-8000-000000000001")
P2 = UUID("d8a3f9b0-1111-4000-8000-000000000002")


def _customer(n: int) -> CustomerType:
    if n == 7:
        return CustomerType(customer_id="cust-07", fullname="Testaccount", shortcode="TA")
    return CustomerType(customer_id=f"cust-{n:02d}", fullname=f"Customer {n:02d}", shortcode=f"C{n:02d}")


CUSTOMERS = [_customer(n) for n in range(30)]


async def resolve_customers(
    info: OrchestratorInfo,
    filter_by: list[GraphqlFilter] | None = None,
    sort_by: list[GraphqlSort] | None = None,
    first: int = 10,
    after: int = 0,
) -> Connection[CustomerType]:
    """Thirty customers, as a deployment's CRM-backed resolver would return them (core's own knows one)."""
    return to_graphql_result_page(CUSTOMERS[after : after + first + 1], first, after, len(CUSTOMERS))


@strawberry.type(description="Orchestrator queries")
class DemoQuery(Query):
    customers: Connection[CustomerType] = authenticated_field(resolver=resolve_customers, description="Customers")


def widget_demo_form() -> object:
    class WidgetDemoPage(FormPage):
        customer_id: CustomerId
        product_id: product_id([P1, P2])  # type: ignore[valid-type]
        customers: list[CustomerId] = []
        note: str = ""

    user_input = yield WidgetDemoPage
    return user_input.model_dump()


@step("Echo the customer")
def echo(customer_id: str) -> dict:
    return {"echoed": customer_id}


@workflow(initial_input_form=widget_demo_form, target=Target.SYSTEM)
def widget_demo() -> StepList:
    return init >> echo >> done


LazyWorkflowInstance("demo", "widget_demo")
```

- [ ] **Step 3: The app and CLI** — create `tests/integration/core/wsgi.py`:

```python
"""orchestrator-core with the widget demo: its customers resolver, its task, and the MCP server (MCP_ENABLED)."""

import demo
from orchestrator.core import OrchestratorCore
from orchestrator.core.settings import AppSettings

app = OrchestratorCore(base_settings=AppSettings())
app.register_graphql(query=demo.DemoQuery)
```

and `tests/integration/core/main.py`:

```python
"""orchestrator-core CLI for the widget stack (migrations)."""

import demo  # noqa: F401  registers the task
from orchestrator.core import app_settings
from orchestrator.core.cli.main import app as core_cli
from orchestrator.core.db import init_database

if __name__ == "__main__":
    init_database(app_settings)
    core_cli()
```

- [ ] **Step 4: Seed products and the task's row** — create `tests/integration/core/seed.py`:

```python
"""The rows core needs for the widget demo: three products, and the task (a process needs its workflows row)."""

from sqlalchemy import select

from orchestrator.core import app_settings
from orchestrator.core.db import ProductTable, WorkflowTable, db, init_database

PRODUCTS = [
    ("d8a3f9b0-1111-4000-8000-000000000001", "Widget Port 10G"),
    ("d8a3f9b0-1111-4000-8000-000000000002", "Widget Port 1G"),
    ("d8a3f9b0-1111-4000-8000-000000000003", "Widget Node"),
]


def seed() -> None:
    if db.session.scalars(select(WorkflowTable).filter(WorkflowTable.name == "widget_demo")).first() is not None:
        print("seed: already present, skipping")
        return
    db.session.add_all(
        [
            *(
                ProductTable(product_id=pid, name=name, description=name, product_type="Widget", tag="WIDGET", status="active")
                for pid, name in PRODUCTS
            ),
            WorkflowTable(name="widget_demo", target="SYSTEM", description="Widget demo", is_task=True),
        ]
    )
    db.session.commit()
    print("seed: created 3 products and the widget_demo task")


if __name__ == "__main__":
    init_database(app_settings)
    seed()
```

- [ ] **Step 5: The entrypoint** — create `tests/integration/core/entrypoint.sh` (mode `755`):

```bash
#!/bin/bash
# Widget integration stack: migrate, seed, serve.
set -eu

cd /home/orchestrator
export PATH="/home/orchestrator/.venv/bin:$PATH"

# The image ships orchestrator-core without the [mcp] extra; install it for the version the image carries.
if ! python -c "import fastmcp" 2>/dev/null; then
    core_version=$(python -c "from importlib.metadata import version; print(version('orchestrator-core'))")
    uv pip install --python /home/orchestrator/.venv "orchestrator-core[mcp]==${core_version}"
fi

if [ ! -f alembic.ini ]; then
    python main.py db init
fi
python main.py db upgrade heads
python seed.py
exec uvicorn --host 0.0.0.0 --port 8080 wsgi:app
```

- [ ] **Step 6: The compose file** — create `tests/integration/docker-compose.yml`:

```yaml
# orchestrator-core for the form-fill widget integration tests:
#   docker compose -f tests/integration/docker-compose.yml up -d --wait
#   WFO_INTEGRATION_CORE_URL=http://localhost:8087 uv run pytest -m integration
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: nwa
      POSTGRES_PASSWORD: nwa
      POSTGRES_DB: orchestrator-core
    healthcheck:
      test: ["CMD", "pg_isready", "--username", "nwa"]
      interval: 3s
      timeout: 3s
      retries: 20

  orchestrator:
    image: ${CORE_IMAGE:-ghcr.io/workfloworchestrator/orchestrator-core:5.4.0}
    environment:
      DATABASE_URI: postgresql+psycopg://nwa:nwa@postgres/orchestrator-core
      MCP_ENABLED: "True"
      OAUTH2_ACTIVE: "False"
      TESTING: "False"
      EXECUTOR: "threadpool"
    volumes:
      - ./core/wsgi.py:/home/orchestrator/wsgi.py:ro
      - ./core/main.py:/home/orchestrator/main.py:ro
      - ./core/demo.py:/home/orchestrator/demo.py:ro
      - ./core/seed.py:/home/orchestrator/seed.py:ro
      - ./core/entrypoint.sh:/home/orchestrator/entrypoint.sh:ro
    entrypoint: ["/home/orchestrator/entrypoint.sh"]
    ports:
      - "8087:8080"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health/')"]
      start_period: 30s
      interval: 5s
      timeout: 5s
      retries: 30
```

- [ ] **Step 7: Bring it up and check it answers**

Run:
```bash
chmod +x tests/integration/core/entrypoint.sh
docker compose -f tests/integration/docker-compose.yml up -d --wait --wait-timeout 300
curl -s -X POST http://localhost:8087/api/graphql -H 'content-type: application/json' \
  -d '{"query":"{ customers(first: 100) { page { customerId fullname shortcode } } }"}' | head -c 300
curl -s http://localhost:8087/api/products/ | python -c "import json,sys; print([p['name'] for p in json.load(sys.stdin)])"
```
Expected: the GraphQL answer starts `{"data":{"customers":{"page":[{"customerId":"cust-00","fullname":"Customer 00","shortcode":"C00"}` and the products list contains `Widget Port 10G`, `Widget Port 1G`, `Widget Node`.

If the GraphQL call answers with an `errors` entry about authorization, add `OAUTH2_AUTHORIZATION_ACTIVE: "False"` to the orchestrator's environment and bring the stack up again. If `demo` cannot be imported by `LazyWorkflowInstance`, set `PYTHONPATH: /home/orchestrator` in the environment.

- [ ] **Step 8: Commit**

```bash
git add tests/integration pyproject.toml
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Add an orchestrator-core stack for form-fill widget integration tests"
```

---

### Task 9: Integration tests of the built-ins against the real core

**Files:**
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_widgets_integration.py`

**Interfaces:**
- Consumes: the stack and its data (Task 8, verbatim); `CustomerIdWidget`, `ProductIdWidget` (Task 3); `CoreGraphQL` (Task 2); `enrich` (Task 4); `FormFillSkill(widgets=, graphql=, choose=)` (Task 6); `open_form`, `turn` (existing `tests/test_form_fill.py`).
- Produces: nothing for later tasks.

- [ ] **Step 1: The fixtures** — create `tests/integration/conftest.py`:

```python
"""The orchestrator-core of ``tests/integration/docker-compose.yml``; every test here is skipped without it."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("OAUTH2_OUTBOUND_ACTIVE", "False")  # the stack runs authless: no service token to fetch

import pytest
from pydantic_ai.mcp import MCPToolset

from orchestrator_agent.form_fill.widgets import CoreGraphQL, WidgetContext

CORE_URL = os.environ.get("WFO_INTEGRATION_CORE_URL")


def pytest_collection_modifyitems(config, items):
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.integration)
            if not CORE_URL:
                item.add_marker(pytest.mark.skip(reason="WFO_INTEGRATION_CORE_URL is not set"))


@pytest.fixture
def core() -> MCPToolset:
    return MCPToolset(f"{CORE_URL}/mcp")


@pytest.fixture
def graphql() -> CoreGraphQL:
    return CoreGraphQL(f"{CORE_URL}/api/graphql")


@pytest.fixture
def ctx(core, graphql) -> WidgetContext:
    return WidgetContext(call_tool=core.direct_call_tool, graphql=graphql)
```

- [ ] **Step 2: The tests** — create `tests/integration/test_widgets_integration.py`:

```python
"""The built-in widgets and the skill against a real orchestrator-core: its GraphQL schema, its MCP tools, its forms."""

from __future__ import annotations

import pytest

from orchestrator_agent.form_fill.core_bridge import choices, page_model, widget_mark
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.form_fill.widgets import BUILTIN_WIDGETS, CoreGraphQL, CustomerIdWidget, ProductIdWidget
from orchestrator_agent.form_fill.widgets.enrich import enrich
from orchestrator_agent.state import Decision, SearchState

from ..test_form_fill import open_form, turn

P1, P2 = "d8a3f9b0-1111-4000-8000-000000000001", "d8a3f9b0-1111-4000-8000-000000000002"
KEY = "widget_demo"


class Chooser:
    def __init__(self, picks):
        self.picks, self.calls = picks, []

    async def choose(self, title, options, words):
        self.calls.append([value for value, _ in options])
        return self.picks


class Refuses:
    async def choose(self, title, options, words):
        raise AssertionError("no model call expected")


def skill(graphql, choose=None) -> FormFillSkill:
    return FormFillSkill(widgets=list(BUILTIN_WIDGETS), graphql=graphql, choose=choose)


async def test_customers_come_from_cores_graphql_schema(ctx):
    options = await CustomerIdWidget().options({"type": "string", "format": "customerId"}, ctx)
    assert len(options) == 30 and options[0].value == "cust-00" and options[0].label == "Customer 00 (C00)"


@pytest.mark.parametrize(
    "words,first",
    [
        pytest.param("Testaccount", "cust-07", id="name"),
        pytest.param("C12", "cust-12", id="shortcode"),
        pytest.param("customer 02", "cust-02", id="most-words-first"),
        pytest.param("nobody", None, id="nothing"),
    ],
)
async def test_searching_the_customers(ctx, words, first):
    options = await CustomerIdWidget().options({"type": "string", "format": "customerId"}, ctx, search=words)
    assert (options[0].value if options else None) == first


async def test_products_come_from_list_products_restricted_to_the_fields_ids(ctx):
    field = {"type": "string", "format": "productId", "extraProperties": {"productIds": [P1, P2]}}
    options = await ProductIdWidget().options(field, ctx)
    assert sorted(option.label for option in options) == ["Widget Port 10G", "Widget Port 1G"]


async def test_the_real_page_is_enriched(core, ctx):
    page = await core.direct_call_tool("get_workflow_form", {"workflow_key": KEY, "page_inputs": []})
    enriched = await enrich(page["schema"], BUILTIN_WIDGETS, ctx)
    fields = page_model(enriched.schema).model_fields
    assert sorted(choices(fields["product_id"])) == sorted([P1, P2])  # two products: chips
    assert widget_mark(fields["customer_id"]) == {"id": "customerId", "total": 30}  # thirty customers: typed
    assert set(enriched.long_lists) == {"customer_id", "customers"}
    assert widget_mark(fields["note"]) is None


async def test_an_exact_name_walks_to_the_approval_and_core_runs_it(core, graphql):
    state, form = SearchState(), skill(graphql, Refuses())
    reply = await open_form(form, core.direct_call_tool, state, KEY, {"customer_id": "testaccount", "product_id": P1})
    assert reply.status == "confirming" and reply.values["customer_id"] == "cust-07"
    assert reply.labels["customer_id"] == "Testaccount (TA)" and reply.labels["product_id"] == "Widget Port 10G"
    started = await turn(form, core.direct_call_tool, state, Decision.START)  # the human's approval
    assert started.status == "started" and started.process_id


async def test_the_chooser_picks_among_the_candidates(core, graphql):
    chooser = Chooser(["cust-12"])
    reply = await open_form(skill(graphql, chooser), core.direct_call_tool, SearchState(), KEY, {"customer_id": "twelve", "product_id": P1})
    assert reply.values["customer_id"] == "cust-12" and len(chooser.calls[0]) == 30  # all thirty: a short list


async def test_several_fits_are_asked_again_as_chips(core, graphql):
    chooser = Chooser(["cust-01", "cust-02"])
    reply = await open_form(skill(graphql, chooser), core.direct_call_tool, SearchState(), KEY, {"customer_id": "one or two", "product_id": P1})
    customer = reply.question("customer_id")
    assert reply.status == "gathering" and customer.values == ("cust-01", "cust-02")


async def test_a_list_of_customers_resolves_each_item(core, graphql):
    answers = {"customer_id": "C03", "customers": "testaccount, C05", "product_id": P2}
    reply = await open_form(skill(graphql, Refuses()), core.direct_call_tool, SearchState(), KEY, answers)
    assert reply.values["customers"] == ["cust-07", "cust-05"]


async def test_unreachable_graphql_leaves_customer_free_text(core):
    unreachable = CoreGraphQL("http://127.0.0.1:9/api/graphql", timeout=2)
    reply = await open_form(skill(unreachable), core.direct_call_tool, SearchState(), KEY, {"product_id": P1})
    customer = reply.question("customer_id")
    assert customer.choices == () and customer.hint == ""  # as before widgets: typed, nothing resolved
```

- [ ] **Step 3: Run against the stack**

Run:
```bash
docker compose -f tests/integration/docker-compose.yml up -d --wait --wait-timeout 300
WFO_INTEGRATION_CORE_URL=http://localhost:8087 uv run pytest tests/integration -m integration -v
```
Expected: all pass. Then `uv run pytest -q` (no env var): the integration tests are reported as skipped and everything else passes.

If `test_an_exact_name_walks_to_the_approval_and_core_runs_it` fails on `create_workflow` with a 404 "Workflow does not exist", core did not import `demo` for the task: check `docker compose -f tests/integration/docker-compose.yml logs orchestrator` and apply the `PYTHONPATH` note of Task 8 Step 7.

- [ ] **Step 4: Commit**

```bash
uv run ruff check tests/integration && uv run ruff format tests/integration
git add tests/integration
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Add integration tests of the built-in widgets against orchestrator-core"
```

---

### Task 10: CI job and README

**Files:**
- Modify: `.github/workflows/ci.yml` (add the `integration` job)
- Modify: `README.md` (settings table; a "Form field widgets" section)

**Interfaces:**
- Consumes: the stack (Task 8), the tests (Task 9), the settings (Tasks 1–2).
- Produces: nothing for later tasks.

- [ ] **Step 1: The CI job** — in `.github/workflows/ci.yml`, after the `test` job (same indentation), add:

```yaml
  integration:
    name: Integration tests (orchestrator-core)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Install uv and set the python version
        uses: astral-sh/setup-uv@v7
        with:
          version: "0.10.8"
          python-version: "3.13"

      - name: Install the project
        run: uv sync --dev

      - name: Start orchestrator-core
        run: docker compose -f tests/integration/docker-compose.yml up -d --wait --wait-timeout 300

      - name: Run integration tests
        env:
          WFO_INTEGRATION_CORE_URL: http://localhost:8087
        run: uv run pytest tests/integration -m integration -v

      - name: Core logs
        if: failure()
        run: docker compose -f tests/integration/docker-compose.yml logs --tail 200
```

- [ ] **Step 2: Validate the YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: README settings** — add two rows to the Configuration table in `README.md`, after `WFO_CORE_MCP_URL`:

```markdown
| `WFO_CORE_GRAPHQL_URL` | *(derived)* | URL of orchestrator-core's GraphQL API, which form-fill widgets read options from (customers). Unset: `WFO_CORE_MCP_URL` with `/mcp` replaced by `/api/graphql` |
| `FORM_WIDGET_EXTENDER` | *(unset)* | `package.module:callable` that receives the form-fill widgets and returns the list to use (a deployment's own formats first). Unset: the built-ins (`customerId`, `productId`) |
```

- [ ] **Step 4: README section** — add after the LibreChat section of the form-fill documentation (the paragraph that starts `**LibreChat directly (chat completions).**` and its list):

````markdown
**Form field widgets.** Some form fields carry no options in core's schema: core's `CustomerId` is a string
with `format: customerId`, and the WFO frontend's customer select fetches the customers itself. The agent does
the same through *widgets*, its counterpart of pydantic-forms' component matchers. Before a page is asked, each
field a widget matches gets its options from core (GraphQL or MCP, as the person asking): up to ten are asked
as options to pick; a longer list is typed, and what is typed is resolved to one of its options — an exact name
directly, otherwise the agent's model reads the words over the candidates — before core sees it. Words that fit
several options are asked again as those options; words that fit none are asked again. Typed text is never sent
to core for such a field.

The agent ships widgets for the formats orchestrator-core defines (`customerId`, `productId`). A deployment adds
its own formats — subscriptions, ports, contacts — from its own package, like the frontend's
`componentMatcherExtender`:

```python
# my_widgets.py — FORM_WIDGET_EXTENDER=my_widgets:extend
from orchestrator_agent.form_fill.widgets import Option, Widget


class LocationWidget(Widget):
    id = "locationCode"

    def matches(self, field):
        return field.get("format") == "locationCode"

    async def fetch(self, field, ctx):
        data = await ctx.graphql("query { locations { code name } }")
        return [Option(row["code"], row["name"]) for row in data["locations"]]


def extend(widgets):
    return [LocationWidget(), *widgets]  # the first widget that matches a field is its widget
```

`ctx.values` holds the form values known so far (a field that depends on another), `ctx.call_tool` calls
core's MCP tools, and `orchestrator_agent.form_fill.widgets.core_auth()` authenticates a widget's own
`httpx` client to core's REST endpoints. A widget whose source fails leaves its field typed, as without it.
````

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml README.md
PATH="$HOME/.local/bin:$PATH" SKIP=mypy git commit -m "Run the widget integration tests in CI and document widgets"
```

- [ ] **Step 6: Full verification** (once, at the end)

Run:
```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```
Expected: all tests pass (integration tests skipped without the env var), ruff clean, mypy clean.
