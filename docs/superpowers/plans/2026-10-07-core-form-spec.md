# An agent-facing form page in orchestrator-core — Plan

**Date:** 2026-10-07
**Status:** Proposed (research done against orchestrator-core `main` @ 5.4.0, orchestrator-agent `main` @ 1389918,
pydantic-forms 2.6.0, fastmcp 3.2.4, pydantic-ai-slim 2.40.0).
**Closes:** follow-ups 5 and 6 of [the form-fill plan](2026-09-23-workflow-form-fill.md).

**Goal:** Let orchestrator-core serve its workflow forms in a shape an agent can use as data, so that
orchestrator-agent can delete `form_fill/core_bridge.py` and the other code that undoes core's
browser-oriented output, without losing any form-fill behaviour.

---

## 1. What `core_bridge.py` does today, and why it exists

Core's MCP tool `get_workflow_form(workflow_key, page_inputs)` re-runs a workflow's form generator with
the pages so far and returns the *next* page as the JSON schema pydantic-forms generates for the
browser UI (`WorkflowFormPage{page, complete, schema}`,
`orchestrator/core/api/api_v1/endpoints/mcp_tools.py:154`). A page core rejects is not a result at all:
`FormValidationError` becomes an HTTP 400 whose JSON body (`{type, detail, title, validation_errors,
status}`, pydantic-forms' `form_error_handler`) fastmcp folds into a tool *error* as its Python repr:
`"HTTP error 400: Bad Request - {'type': 'FormValidationError', ...}"`
(`fastmcp/server/providers/openapi/components.py:238`). pydantic-ai raises that as `ModelRetry`.

`core_bridge.py` (348 lines) undoes both on the agent side. It has two halves:

| Half | Functions | What it undoes |
| --- | --- | --- |
| **Schema reading** (goes) | `page_model`, `form_model`, `_object_model`, `_field`, `_annotation`, `_allowed`, `_labels`, `_no_options`, `resolve_property`, `summaries`, `is_read_only` | `$ref` / single-`allOf` / nullable-`anyOf` indirection, `$defs` for nested models, `format` markers that mean "display only", `uniforms`/`extraProperties.disabled` for read-only, enum labels in the `options` side table, `const` as a one-option choice, `extraProperties.data` for summary tables, `minItems`/`maxItems` on lists |
| **Error reading** (goes) | `form_errors`, `_FormErrorBody` | `ast.literal_eval` of the dict repr inside the tool error text |
| **Model reading** (stays, shrinks) | `value_type`, `is_list`, `item_type`, `choices`, `labels`, `label_of`, `is_accept`, `item_bounds`, `is_single_pick` | Nothing of core's: these read the pydantic model the first half built. With a typed field spec from core, the skill reads the spec directly and only the interpreter still needs a pydantic model |

Around it, the same browser-shape leaks into:

- `state.py`: `FormError` and `SummaryTable` are hand-copied `TypedDict`s of pydantic-forms' `ErrorDict`
  and `SummaryData` (pydantic-forms' own cannot be imported: `pydantic_forms.validators` pulls in a
  contact-person field that needs `email-validator`, which the agent does not ship). `ACCEPT_VALUE` is
  pydantic-forms vocabulary spelled out again.
- `FormFillSession.pages` persists raw page schemas (`list[dict]`) and rebuilds the models every turn.
- `skill.py`: `questions()`/`question()`/`_see_page()`/`_labels()`/`_summary()` read `FieldInfo`s;
  `_walk()` and `_start()` catch `ModelRetry` and text-parse it; `workflows()` lists the catalogue in two
  calls because `list_workflows` rejects a call without arguments (HTTP 422 `body: Field required`:
  FastAPI treats the all-optional `ListWorkflowsRequest` body as required).
- `tests/test_form_fill.py` (52 tests) fakes core with raw JSON schemas "as core 5.4 renders it".

Core has the lossless source: the pydantic model class of each page (`type[FormPage]`), with `frozen`
for display-only fields, `Choice` members with their labels, `Annotated` constraints, nested models and
`default_factory`. Only the model's *JSON schema* leaves core today, and only the error's *text*.

## 2. What to build in orchestrator-core

### 2.1 The contract: a field spec, and a rejected page as a result

New response models in `orchestrator/core/schemas/mcp_tools.py` (same `OrchestratorBaseModel` style as
the rest of the file):

```python
class FormFieldOption(OrchestratorBaseModel):
    value: Any          # what the caller submits
    label: str          # how a person sees it (a product's name, a Choice label)


class FormFieldError(OrchestratorBaseModel):
    loc: list[int | str]  # path of the field on the rejected page; ["__root__"] for a page-level error
    msg: str              # pydantic-forms' translated message, identical to what the UI shows
    type: str             # pydantic's error type: "missing", "enum", "value_error", ...


class FormField(OrchestratorBaseModel):
    name: str
    title: str
    description: str | None = None
    kind: Literal["string", "integer", "number", "boolean", "list", "object", "any"]
    format: str | None = None     # the pydantic-forms / core marker: accept, productId, subscription, long, timestamp, customerId, ...
    required: bool
    default: Any = None           # default_factory evaluated, as GenerateFormJsonSchema does for the UI
    nullable: bool = False
    options: list[FormFieldOption] | None = None  # Choice / Literal / const / Accept; [] = a choice with no option today; None = free
    item: "FormField | None" = None               # kind == "list": the item's shape (its options, fields, ...)
    min_items: int | None = None
    max_items: int | None = None
    unique_items: bool = False
    fields: "list[FormField] | None" = None       # kind == "object": the nested model's fields, in order
    read_only: bool = False       # read_only_field / read_only_list: shown, fixed to ``default``, never asked
    display_only: bool = False    # label, divider, hidden, markdown, callout, summary, subscription: never submitted
    data: Any = None              # what a display field shows: summary tables (SummaryData), markdown/callout text, Accept items
```

`WorkflowFormPage` is extended in place (no second tool: one tool with two shapes confuses callers):

```python
class WorkflowFormPage(OrchestratorBaseModel):
    page: int                                   # the page this result is about
    complete: bool                              # kept
    status: Literal["next", "complete", "rejected"]
    title: str | None = None                    # None for pydantic-forms' "unknown" placeholder
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")   # kept: the browser schema
    fields: list[FormField] | None = None       # NEW: the page as data; None when complete
    errors: list[FormFieldError] = Field(default_factory=list)  # rejected: core's verdict on page ``page``
```

**The one behaviour change:** a rejected page is returned as `status="rejected"` with `errors` and the
rejected page's `fields`, instead of raising HTTP 400 into a tool error. For a direct LLM caller this
is at least as good (structured errors; `complete` stays false; the docstring says what to do), and
`create_workflow` still rejects bad pages with 400, so nothing can start on an ignored verdict. Every
other failure (unknown workflow 404, too many pages 422) stays an error.

Everything the agent reads today maps onto the spec:

| Agent today | Spec |
| --- | --- |
| `page_model` fields | `fields` where `not display_only and not read_only` |
| `choices(info)`, `_no_options` | `options` (`item.options` for a list); `[]` keeps "no option today" |
| `labels`, `label_of` | `options[].label` |
| `is_accept` | `format == "accept"` (its one option's value is what consent sends) |
| `is_list`, `item_bounds`, `is_single_pick` | `kind == "list"`, `min_items`/`max_items`; single pick = `options is not None and (kind != "list" or max_items == 1)` |
| `value_type is bool` | `kind == "boolean"` |
| `info.is_required()`, `info.default`, `info.title` | `required`, `default`, `title` |
| `summaries(schema)` | `[f.data for f in fields if f.format == "summary"]` |
| nested `_object_model` | `fields` / `item.fields` |
| `form_errors(text)` | `errors` |
| `schema["title"]` | `title` |

### 2.2 The walk: core's own page loop

pydantic-forms' `post_form` (sync, `pydantic_forms/core/sync.py`) only ever hands out the next page's
JSON schema (inside `FormNotCompleteError`) and raises `FormValidationError` without saying which page
failed. To build the spec from the model class and to name the rejected page, core needs the loop
itself. New module `orchestrator/core/forms/walk.py`:

```python
@dataclass(frozen=True)
class NextPage:   model: type[FormPage]; index: int
@dataclass(frozen=True)
class Complete:   state: State
@dataclass(frozen=True)
class Rejected:   model: type[FormPage]; index: int; error: FormValidationError

def walk_form(form_generator, state, user_inputs, locale="en_US") -> NextPage | Complete | Rejected: ...
```

It is the same ~25 lines as `post_form`: init the generator, `send` each validated page, and on
`ValidationError` build `FormValidationError(model.__name__, e, PydanticI18n(translations), locale)`
exactly as pydantic-forms does, so messages are identical to the UI's. `FormOverflowError` (more inputs
than pages) is raised as today. The copy is deliberate: it avoids waiting on an upstream pydantic-forms
release; file a pydantic-forms issue to expose the page model and the rejected index from `post_form`
so the copy can go later (see Options, C).

### 2.3 The builder: model class → `list[FormField]`

New module `orchestrator/core/forms/spec.py`, `form_fields(model: type[BaseModel]) -> list[FormField]`,
one `FormField` per `model_fields` entry in order. Per `FieldInfo`:

- `display_only`: `info.frozen` (Label, Divider, Hidden, Markdown, Callout, MigrationSummary,
  DisplaySubscription are all `Field(frozen=True)`) — exact, where the agent could only match a set of
  `format` names.
- `read_only`: `extraProperties.disabled` (or the deprecated `uniforms.disabled`) in
  `json_schema_extra` (`read_only_field`, `read_only_list`).
- `format` and `data`: from `json_schema_extra` — a dict, or a callable to call with an empty dict as
  core's `summary_form._field_format` already does (Markdown, Callout, MigrationSummary build their
  schema that way). `Accept` and `Choice` set theirs on the *type*, so: `issubclass(t, Accept)` →
  `format="accept"`, `options=[ACCEPTED]`, `data=cls.data`; `issubclass(t, Choice)` →
  `options=[(m.value, m.label) for m in cls.__members__.values()]`; any other `Enum` → value/label equal;
  `Literal[...]` → its values.
- `kind` and shape: unwrap `Optional` (→ `nullable`), `Annotated` metadata (`MinLen`/`MaxLen` →
  `min_items`/`max_items`; `uniqueItems` in extra → `unique_items`), `list[X]` → `kind="list"` with
  `item=form_field(X)`, a `BaseModel` → `kind="object"` with `fields=form_fields(X)`, then
  `str`/`UUID`/`datetime`/`Enum` → string, `int` → integer, `float` → number, `bool` → boolean, else any.
- `default`: `info.default`, or `info.default_factory()` when it takes no data (same rule and same
  guard as pydantic-forms' `GenerateFormJsonSchema.default_schema`).
- `required`: `info.is_required()`; `title`: `info.title` or pydantic's own humanisation of the name;
  `description`: `info.description`.

Core can test this against the real field types (it ships `pydantic_forms.validators`), which the agent
never could.

### 2.4 Endpoint changes (`mcp_tools.py`)

- `get_workflow_form_endpoint`: run `walk_form` in `run_in_threadpool` (form generators use the sync
  `db.session`; today the sync `generate_form` blocks the event loop), add `reporter` (from
  `Depends(user_name)`) to the initial state so the agent-facing walk sees the same state as
  `start_process`, and return `status`/`title`/`fields`/`errors` next to the kept `schema`. Update the
  docstring: "stop when `status` is `rejected`: fix the fields in `errors` and resubmit that page".
- `list_workflows_endpoint`: `params: ListWorkflowsRequest = Body(default_factory=ListWorkflowsRequest)`
  so a call without arguments lists everything, as the description already promises (follow-up 6).
- Optional, cheap, enables follow-up 2 later: `get_process_status` adds `form_fields` for a suspended
  process, from the same builder (`enrich_process` already generates that form).

### 2.5 Tests, docs, release

- `test/unit_tests/forms/test_spec.py`: one parametrized case per field type → expected `FormField`:
  `Accept`, `Choice` with and without labels, `Choice` with no members, `choice_list` with bounds,
  `unique_conlist`, `ListOfOne`/`ListOfTwo`, `read_only_field`, `read_only_list`, `Label`/`Divider`/`Hidden`,
  `Markdown`/`Callout`/`MigrationSummary` (with `data`), `DisplaySubscription`, `LongText`, `Timestamp`,
  `ProductId`, `CustomerId`, `Optional[str]`, `Literal`, nested `BaseModel`, `list[BaseModel]`,
  `default_factory`, `bool`/`int`/`float`/`UUID`/`datetime`. Plus a golden test over core's injected
  first pages (`NewProductPage` with product labels, `ModifySubscriptionPage`) and a summary-form page.
- `test/unit_tests/forms/test_walk.py`: a three-page generator → `NextPage` at 0 and 1, `Complete` at 3,
  `Rejected` at page 1 with `loc`/`msg`/`type`, a page-level (`__root__`) error, overflow.
- `test/unit_tests/mcp/test_mcp.py`: through the fastmcp `Client`, a registered dummy workflow:
  `get_workflow_form` returns `fields`; a bad page comes back as a *result* with `status="rejected"`
  (not a `ToolError`); `list_workflows` with `{}` succeeds. Keep `EXPECTED_TOOL_NAMES` unchanged.
- Docs: `docs/reference-docs/mcp.md` — tool table row and a short "Form pages for agents" subsection
  with the `FormField` fields; CHANGELOG entry naming the rejected-page change.
- Release: next minor (5.5.0). The agent pins `orchestrator-core>=5.5.0`.

## 3. What then goes from orchestrator-agent (second PR, after the core release)

1. **`form_fill/core_bridge.py` is deleted.** What survives is one small builder, `form_fill/model.py`
   (~60 lines): `page_model(fields) -> type[BaseModel]` and `form_model(pages)`, built from `FormField`
   (`kind` → type, `options` → `Literal`, `fields` → nested model, `required`/`default`/`title`/
   `description`), used only by `interpret.py` for the model's output type and its JSON schema.
2. **`skill.py` reads the spec.** `questions`/`question`/`_see_page`/`_labels`/`_summary` take
   `FormField`s (mapping table above). `_walk` handles `status == "rejected"` as a value; the
   `ModelRetry` branch keeps only "no page submitted → core refuses the workflow → failed". `_start`:
   on a `create_workflow` failure, re-call `get_workflow_form` with the same pages — `rejected` reopens
   the form at that page with core's structured verdict, `complete` means it was no verdict on the
   pages (lock, predicate, auth, a race) and the form closes with the error as reason, as today. One
   extra read on a rare path replaces `form_errors`. `workflows()` becomes one call.
3. **`state.py`:** `FormError`, `SummaryTable`, `ACCEPT_VALUE` go; `FormReply.rejected:
   list[FormFieldError]`, `FormReply.summary: list[SummaryData]`-shaped as before (the summary field's
   `data`); `FormFillSession.pages: list[WorkflowFormPage]`. `SUBSCRIPTION_ID` stays (the handoff).
4. **`WriteToolGate` keys on core's annotations** instead of names: pydantic-ai 2.40 puts each MCP
   tool's `annotations` on `ToolDefinition.metadata` (`pydantic_ai/mcp.py:1309`), so hide every tool with
   `readOnlyHint is False`. `WRITE_TOOL_NAMES` goes; the other `tool_names.py` constants stay (plugin
   ownership, prompts, the startup contract check). No core work: the hints are already declared per
   route.
5. **Tests:** fixtures in `test_form_fill.py` / `test_form_interpret.py` become `FormField` specs
   (shorter than the schemas they replace); the ~15 bridge tests go; add one contract test that feeds a
   page recorded from core 5.5 (the eval stack's demo products) through the skill end to end.
6. **Docs:** README "One pydantic model per page" paragraph rewritten; follow-ups 5 and 6 of the
   form-fill plan marked done.

Expected size: `form_fill/` 1,340 → ~1,000 lines, `state.py` −30, and the agent no longer contains any
knowledge of how pydantic-forms renders a browser form.

## 4. What stays in the agent, and why

- **The walk policy** (`_walk`, `_see_page`, consents, asking an optional-only page once, the interpreter
  loop): these are UX decisions of one agent, not facts about a form. Core stays a lossless form server.
- **The pydantic model for the interpreter**: the LLM's output type must be a model; building it from a
  typed spec is ~60 lines and has no pydantic-forms knowledge in it.
- **Persistence** (`persistence.py` writing `graph_snapshots`/`agent_runs` through core's SQLAlchemy
  models) and the `orchestrator-core` *package* dependency it implies (`DATABASE_URI`, `init_database`).
  Putting that behind a core API is a separate decision with its own PRs; it is what makes the agent a
  fat client, but it is unrelated to forms.

## 5. Options considered

- **A. Contract only (this plan, recommended).** Core returns the page as data and the verdict as a
  result; the agent keeps the walk. Smallest change on both sides, no policy moves into core, and the
  contract is reusable by any other agent and by the suspended-process path.
- **B. Core walks too:** a `fill_workflow_form(workflow_key, values)` tool that applies a flat value map
  page by page and returns the first page it cannot complete plus the validated `page_inputs`. Trims
  `skill.py` further, but moves single-option auto-pick, consent scoping and "ask once" into core as
  product behaviour, and a second agent would want them different. Revisit when there is a second agent.
- **C. Upstream the loop into pydantic-forms** (`FormNotCompleteError.model`, a page index on
  `FormValidationError`) and have core use it instead of its own `walk_form`. Cleaner ownership, but
  gated on a pydantic-forms release; do it after A, then delete `walk.py`.
- **D. Keep the 400, make the error body machine-readable** (JSON instead of a Python repr). Still text
  parsing of a tool error, and fastmcp's error formatting is not configurable per tool. Rejected.

## 6. Sequence and verification

1. Agree the contract (section 2.1) with whoever maintains direct LLM use of `get_workflow_form`.
2. Core PR: walk + builder + schemas + endpoint + tests + docs. While it is in review, develop the
   agent change against the branch with a `[tool.uv.sources]` path override.
3. Core release; agent PR bumps the pin and lands the trim; `uv run pytest`, `mypy`, `ruff`.
4. Live check as the original plan did: the demo app's create/modify/terminate forms and a summary-form
   workflow through LibreChat and kagent — every stop still shows the same questions, chips, labels,
   summary tables and approval payload as before the change.
