# Form Field Widgets — Design

**Date:** 2026-10-07
**Status:** Approved (pending spec review)

## Goal

Let the form-fill skill handle the pydantic-forms field components whose options are *not* in the JSON
schema core returns, the way the WFO frontend does: a field marked `format: customerId` becomes a pick from
the real customers, not a text box that accepts anything.

The mechanism mirrors pydantic-forms' frontend component registry (`PydanticComponentMatcher` /
`ComponentMatcherExtender`): the agent ships widgets for the formats orchestrator-core itself defines, and a
deployment adds its own (SURF: `subscriptionId`, `imsPortId`, contact persons, …) from a separate package
through an extender, as `surfPydanticFormComponentMatcherExtender` does in the SURF UI.

Success: in the recorded LibreChat flow (`create_service_port`, "10G SP for Testaccount"), the customer is
picked from the real list (typed text resolved to one of its options, a short list shown as buttons), and a
typed "TESTACCOUNT" can no longer travel to core as a customer id.

## Background

### What the recording shows

LibreChat's ask-user card shows `ProductChoice` and `port_mode` as buttons and `customer_id` / `port_id` as
"Type your answer…". The difference is in core's page schema, not in the agent:

| Field | Core's schema (`get_workflow_form`) | Where the WFO UI gets the options |
|---|---|---|
| `product`, `port_mode` | a `Choice`: `enum` + `options` | the schema |
| `customer_id` | `CustomerId = Annotated[str, Field(json_schema_extra={"format": "customerId"})]` | `SurfCustomerSelect` → GraphQL `customers` |
| `port_id` | `ImsPortId`: `int`, `format: imsPortId`, `uniforms` hints | `SurfImsPortIdSelect` → `surf/ims/free_ports/…` |
| `contact_persons` | `list[ContactPerson]` + `customerKey` | `SurfContactNameField` → `surf/crm/contacts/{customer}` |

`core_bridge` types such a field as plain `str`/`int`, `question()` builds an `AskField` without choices,
and both transports render free text. Core's `CustomerId` has no validator: any string passes the form and
fails later, in the workflow's steps.

### The frontend's pattern

pydantic-forms renders each field with the first matching `PydanticComponentMatcher`
(`{id, matcher(field) -> bool, ElementMatch}`); `ComponentMatcherExtender` receives the default list and
returns the final one. The SURF UI prepends its own matchers (`customerId`, `locationCode`, `imsNodeId`,
`imsPortId`, `subscriptionId`, `vlan`, contacts, …); each `Element` fetches its own options, reading hints
from the field schema (`uniforms` / `extraProperties`) and, where needed, other form values.

Of these formats, orchestrator-core itself defines two that are backed by data: `customerId` (`CustomerId`)
and `productId` (`ProductId` / `product_id([...])`, hint `productIds`). Everything else is SURF's.

## Design

### 1. Widget interface and registry

New package `src/orchestrator_agent/form_fill/widgets/`.

```python
@dataclass(frozen=True)
class Option:
    value: str | int                  # what is sent to core
    label: str                        # what a person sees
    aliases: tuple[str, ...] = ()     # other names a person may type for it (a shortcode, a short name)

@dataclass(frozen=True)
class WidgetContext:
    call_tool: CallTool               # core's MCP session, with the caller's token (exists)
    graphql: GraphQL                  # core's /api/graphql, with the caller's token (section 3)
    values: Mapping[str, Any]         # form values known so far (customerKey-style dependencies)

class FieldWidget(Protocol):
    id: str
    def matches(self, field: Mapping[str, Any]) -> bool: ...
    async def options(
        self, field: Mapping[str, Any], ctx: WidgetContext, search: str | None = None
    ) -> Sequence[Option] | None: ...

FieldWidgetExtender = Callable[[list[FieldWidget]], list[FieldWidget]]
```

- `field` is the resolved property schema (`core_bridge.resolve_property`), so hints in `uniforms` /
  `extraProperties` are at hand.
- `options(search=None)` returns every option of the field; `options(search=words)` returns at most 50
  candidates the words could mean. `None` means the field depends on a value that is not known yet (e.g.
  contacts before a customer is chosen): it is left out of this stop and asked once its dependency is
  answered.
- A base class `Widget` implements `options` on top of one abstract `fetch(field, ctx)`: no `search` returns
  what `fetch` returned, a `search` narrows it with `narrow_options` (case-insensitive match on label, value
  and aliases — whole tokens first, then substrings — capped at 50). A widget whose source can search on its
  own (thousands of subscriptions) overrides `options` and passes `search` on instead.

**Registry.** `build_widgets(extender) -> list[FieldWidget]` returns `extender(list(BUILTIN_WIDGETS))`, or
the built-ins when no extender is configured. The first widget whose `matches()` is true wins, so an extender
prepends its widgets (as the SURF UI does) or drops a built-in by `id`.

**Matching rules** (applied by the registry before any widget is asked):

- a property that carries `enum` or `const` is never matched — core's own list wins (core's `ProductChoice`
  page, which carries `format: productId` *and* an enum, is left alone);
- display-only and read-only properties never reach widgets (`DISPLAY_ONLY_FORMATS`, `is_read_only`);
- an `array` property is matched on its `items` schema; its options make it a multi-select.

**Built-ins** (`widgets/core.py`) — the formats orchestrator-core defines:

| id | matches | options |
|---|---|---|
| `customerId` | `type: string`, `format: customerId` | GraphQL `customers(first: 1000000) { page { customerId fullname shortcode } }`; label `fullname (shortcode)`, aliases `fullname` and `shortcode` |
| `productId` | `format: productId` | MCP `list_products`; restricted to `productIds` from `extraProperties` / `uniforms` when present; label the product name |

**Loading.** A new setting `FORM_WIDGET_EXTENDER` (`"package.module:callable"`, unset by default) is
resolved once by `build_form_fill_skill()`. A path that does not import, is not callable, or does not return
a list of widgets fails startup, like the tool-contract check. A SURF package ships `surf_agent_widgets:extend`
with the SURF widgets; nothing SURF-specific lives in this repo.

### 2. Enrichment and resolution in the skill

**Enrichment.** In `FormFillSkill._walk`, right after core returns a page and before it is stored and turned
into a page model:

```python
schema = await enrich(page.schema_, self.widgets, ctx)
```

`enrich` resolves every property (and the `items` of an array) the registry matches and rewrites it:

| Widget result | Written into the property | How the field is asked |
|---|---|---|
| ≤ 10 options | `enum` + `options` (core's own shape) + `x-widget: {id}` | chips / buttons; multi-select for an array |
| > 10 options | `x-widget: {id, total}` (no enum) | free text, with the hint "type a name — n options" |
| `None` | `x-widget: {id, later: true}` | not asked on this stop |
| the widget raised | unchanged (the error is logged with the widget id) | as today: free text |

Ten keeps a question within LibreChat's twelve options with "Keep default" added. Everything downstream of
`enum` + `options` already works: the `Literal` field type, chips in both transports, multi-select, the
single-option rule in `_see_page`, labels in the summary and the approval, the labels in the interpreter's
schema. `core_bridge._field` carries `x-widget` into the field's `json_schema_extra`.

`enrich` memoizes `options()` per walk on `(widget id, hint fields of the property, dependency values)`; a
page regenerated in the same walk does not fetch twice. Nothing is cached across turns: option lists depend on
the user and change (free ports), and the UI does not cache them either.

**Resolution of a long-list answer.** What a person typed for a long-list field is words, never a value.
Before the page is submitted to core, `_walk` hands each such answer to `resolve_choice` (collecting a
page's values becomes async for this):

1. **Exact** — a case-insensitive match of the words on exactly one option's label, value or alias is that
   option; no model call ("TESTACCOUNT" is the alias of "Testaccount (TA)").
2. **Candidates** — the full list when it has at most 200 options (abbreviations such as "UT" then still
   resolve), otherwise `widget.options(field, ctx, search=words)`.
3. **Interpreter** — the skill's `Interpreter` reads the words against one field typed as a `Literal` over the
   candidates, with their labels (a new entry point next to `answers`, reusing `fields_model` /
   `reading_type`, so the existing rule holds: every option the words fit).

| Interpreter result | What happens |
|---|---|
| exactly one option fits | that option is the value |
| several fit | the field is asked again as chips of those options ("did you mean …") |
| none fit, or no interpreter is configured and step 1 found nothing | asked again with the hint "nothing matched '…'" |

Guarantees:

- **A closed set.** A value is always one of the options the widget returned; typed text for a widget field
  never reaches core.
- **One reading per answer.** The outcome is kept on the session per field and words
  (`FormFillSession.resolved`), so a re-walk from page 0 costs no model call and gives the same value. New
  words for the field replace it.
- **The human confirms.** The label of a resolved value is shown at the approval, before anything starts —
  the interpreter's existing contract.

A multi-select long-list field resolves each item of its answer on its own (one typed text is split on
commas); an item that does not resolve re-asks the field.

**Changes outside `widgets/`:**

- `state.py`: `AskField.hint` (one line a transport shows under the question); `AskField` gets the
  narrowed `choices` / `values` when a re-ask offers candidates; `FormFillSession.resolved`.
- `skill.py`: `enrich` and `resolve_choice` in `_walk`; `question()` sets `hint` for long-list
  and unmatched fields and skips `later` fields.
- `interpret.py`: `Interpreter.choose(field, candidates, words)` with a `ModelInterpreter` implementation.
- `adapters/chat/librechat.py`: `hint` into the question's `description`; `adapters/a2a/kagent.py`: `hint`
  appended to the question text (kagent's card has no description).
- `form_fill/__init__.py`: `build_form_fill_skill(model, widgets=…)`; `app.py` passes the configured list.

Prefilling a widget field from the opening request ("… for Testaccount") is out of scope (plan doc
follow-up 7): the interpreter reads what is typed into the card.

### 3. Data access and auth

`WidgetContext` gives a widget two channels to core, both authenticated as the caller:

- `call_tool` — the agent's existing MCP session (`bind_outbound_token` per run): `list_products`, and any
  core tool an extender needs.
- `graphql(query, variables) -> dict` — a new small client (`widgets/graphql.py`) that posts to core's
  `/api/graphql` with the MCP client's `_ContextVarBearerAuth` (the forwarded user token, else the service
  client-credentials token, refreshed once on 401/403). GraphQL `errors` raise. Customers come from here:
  core has no MCP or REST tool for them, which is why the UI uses GraphQL too.

New setting `WFO_CORE_GRAPHQL_URL`; when unset it is derived from `WFO_CORE_MCP_URL` (`…/mcp` →
`…/api/graphql`), so existing deployments need no new configuration.

The auth is exported as `orchestrator_agent.form_fill.widgets.core_auth()` (an `httpx.Auth`), so an extender
that reads deployment endpoints (SURF's `surf/ims/*`, `surf/crm/*`) builds its client the same way.

**Failures.** A widget that raises leaves its field as it is without widgets (section 2); the form stays
open. Only an extender that fails to load stops startup (section 1).

### 4. Tests

**Integration** (`tests/integration/`, marker `integration`, skipped unless `WFO_INTEGRATION_CORE_URL` is
set, so `uv run pytest` stays offline). A compose stack `tests/integration/docker-compose.yml`, separate from
the eval stack so the eval dataset is not disturbed; same core image and entrypoint pattern, own ports. Its
core app registers:

- three products, one of them outside the task's `productIds`;
- thirty customers, through a `customers` GraphQL resolver override (bare core knows one default customer;
  SURF overrides the resolver the same way);
- a task `widget_demo` with one page: `customer_id: CustomerId`, `product_id: product_id([p1, p2])`,
  `customers: list[CustomerId]`, `note: str`. A task, because a create workflow's own product page is a core
  enum, which widgets leave alone by design.

| Test | Checks |
|---|---|
| `customerId` widget | the GraphQL query and label shape against core's real schema; `options(search=…)` over the thirty customers (exact name, shortcode, substring, nothing) |
| `productId` widget | `list_products` over MCP, restricted to `productIds` (two of three) |
| `enrich` on the real page | products → `enum` + `options`; customers → long-list `x-widget`; `note` untouched; the result builds a valid `page_model` |
| walk, exact | a typed customer name becomes its id without a model call (an interpreter stub that fails when called); the walk reaches `confirming`; the approval carries the label |
| walk, interpreter | a pydantic-ai `FunctionModel` interpreter receives a `Literal` over the candidates only; one fit → taken, two → asked again as chips |
| walk, multi-select | the `customers` list field, each part resolved |
| core decides | the resolved values pass core's own validation and `create_workflow`; the process completes |
| fallback | an unreachable GraphQL URL leaves `customer_id` a free-text question and the walk still works |

Cases that differ only in data are parametrized.

**Unit** (`tests/test_form_widgets.py`, no network): registry order, extender prepend/remove,
`FORM_WIDGET_EXTENDER` loading (good path, bad path fails); the matching rules (enum never matched, array
items, display-only skipped); the `enrich` thresholds (10 / 11 options, 200 / 201); the resolution table
(exact, one, several, none, kept per words, replaced by new words); the GraphQL client forwards the bound
token and refreshes the service token on 401 (`httpx.MockTransport`); both transports render `hint`.

**CI.** A new `integration` job in `.github/workflows/ci.yml` brings the compose stack up, waits for health,
and runs `uv run pytest -m integration`. The existing test job is unchanged.

## Out of scope

- The SURF widgets themselves (a separate package, built on the extender).
- Prefilling widget fields from the opening request.
- Long core enums (the 33-option picker) — they keep today's free-text + interpreter path.
- A cross-turn cache of option lists.
