# Workflow form-fill over A2A — Implementation Plan

**Date:** 2026-09-23 (plan), 2026-09-24 (final state)
**Status:** Implemented as two stacked branches, verified live on the demo app and on production-shaped workflows against a database clone.
`feat/workflow-form-fill` is the deliverable: the deterministic skill with the literal JSON-object
contract, the model's `start_workflow_form` handoff and kagent's human-in-the-loop extension.
`feat/workflow-form-fill-jev` on top adds Jev as an optional typed decision engine behind `JEV_API_KEY`
(the experiment that started this work; it routes in front of the model and prefills option fields).
The sections after "Tasks" are the log of how the design got here, in order; the later ones supersede
the earlier ones where they differ.

**Goal:** Let a calling agent (kagent, over A2A — the only access pattern that matters) start a WFO
workflow by having this agent walk the workflow's multi-page pydantic-forms input form: take what the
caller states, ask for the rest, get the user's confirmation, and start — using only the MCP tools
orchestrator-core already serves.

**Architecture:** *No LLM in the form loop.* Over A2A the caller is an LLM agent, so this agent never
needs to write prose for a human: code walks the pages from page 0 on every turn, and every stop is a
templated message in a JSON-object contract the caller reads and writes reliably (or, for a kagent
caller, a native pause through the HITL extension). Deciding *which* workflow a request asks for is a
judgment call and is the model's: it reads `list_workflows` and hands off with the `start_workflow_form`
tool; the skill walks the first pages right after the model's run and its reply replaces the model's text.
From then on the skill claims every message of the conversation until the form is started or cancelled.
Routing is never a literal match on workflow keys — callers do not speak in keys. The remaining judgment
calls (which option a sentence names, does a free-form reply approve) sit behind the `SystemOne` protocol;
the reply is data (a JSON object keyed by field name) or the interpreter's reading of the message (values and,
only when stated outright, a decision), with Jev they are
decided with a calibrated confidence and gated by a threshold. The model never sees core's write tools.

This replaces v1 of this plan (an LLM-driven loop guarded by a form-completion hook): the live e2e of v1
showed the model confirming before the form was complete and starting without re-confirming after the last
page — both prompt-adherence failures that a code-owned loop makes impossible.

---

## Background facts (verified against orchestrator-core main @ v5.4.0)

- Core's MCP server (`/mcp`, `MCP_ENABLED=true`) exposes `list_workflows(target?, is_task?)`,
  `get_workflow_form(workflow_key, page_inputs?) -> {page, complete, schema}`,
  `create_workflow(workflow_key, json_data: list[dict])` and `get_subscription_available_workflows`.
  `get_workflow_form` re-runs the form generator with the inputs so far and returns the *next* page: pages
  are dynamic and the count is unknown up front.
- A rejected page (`FormValidationError`, HTTP 400 with field names) surfaces as a tool error, which
  pydantic-ai's `direct_call_tool` raises as `ModelRetry`. `FormPage` is `extra="forbid"`.
- pydantic-forms renders `Choice` as `enum` (+ `options` value→label when they differ), `choice_list` as an
  array of a `$ref`'d enum, `ReadOnlyField` as `const` + `uniforms.disabled`, display widgets as
  `format: label|divider|summary|markdown|hidden`, `Accept` as `format: accept` with enum
  `ACCEPTED|INCOMPLETE`. Core's injected create page 0 is a product enum with product names as labels.
  Modify and terminate workflows start with core's `ModifySubscriptionPage` (a raw `subscription_id`).
- Core's per-subscription listing includes the product's create workflows, always with a reason: create
  workflows are not subscription-scoped and must never be vetoed by one.
- Jev (optional branch): `Choice` (≤255 options), `Noul` (probability); calibrated confidence; no text
  generation; reads literally; an unmentioned `Noul` statement reads as a confident false (hence three-way
  choices).
- kagent ≥ 1.0.0-alpha1 passes a remote agent's `input-required` pause through to the human via the
  `https://kagent.dev/extensions/hitl/v1` extension (see the HITL section); callers without the extension
  get every stop as completed text.

---

## File structure (as on `feat/workflow-form-fill`)

- **`src/orchestrator_agent/form_fill/`** (new package)
  - (no renderer) — every reply is a `FormReply` (`state.py`) serialised as JSON: core's data, nothing
    phrased here; the kagent questions and chips are built in `skill.py` from the page model.
  - `core_bridge.py` — `page_model` (core's browser schema → one pydantic model per page, cached; the
    artifact every stop works from) / `form_model` (the walked pages as one), what a built model says
    about its fields (`choices`, `labels`, `is_accept`, `is_list`, `item_bounds`, ...), and core's
    rejection out of the tool error text; the schema-reading half goes with core follow-up 5.
  - `skill.py` — `FormFillSkill`: `handle` (one caller message), `open` (walk a form the model handed off),
    the walk from page 0 with core as the only judge of a page, confirmation, start; every reply a
    `FormReply`; the kagent questions and chips from the page model.
  - `handoff.py` — the model's `start_workflow_form(workflow_key, subscription_id?)` tool: marks the session
    as opening; the capability walks it on the model request that follows.
  - `capability.py` — `FormFillCapability`, the skill as a pydantic-ai capability: `before_model_request`
    runs the skill on the first request of a run (and the handoff walk after the tool) and skips the
    model with the reply (`SkipModelRequest`); `get_toolset` contributes the handoff tool; core is called
    through the agent's own MCP session. The reply is left on `SearchState.form_reply` (transient).
  - `prefill.py` — the engine seam: `SystemOne` protocol, `plans_for` (fields → typed questions), `gate`
    (answers → values with a confidence threshold), `prefill_page`. Engine-agnostic; the Jev adapter is on
    the stacked branch.
  - `hitl.py` — kagent's HITL wire models; a `Reply` → `ask_user_request` / `tool_approval_request`, the
    human's response → the text contract or an approval decision.
  - `__init__.py` — `build_form_fill_skill()`.
- **`state.py`** — the form-fill data model: `FormReply` (every answer as one JSON object: status, page,
  schema, core's rejected errors, values, defaults, process id, reason), `values_in` (the JSON
  object a reply is), `Reply(text, ask | approval)` (the `FormReply` as text plus the kagent stop),
  `FormFillSession{workflow_key, status, request, values, pages (the last walk's page schemas, from which
  the page models are rebuilt), page_inputs, accepted, hitl_request, task_id, asked, interpreted}` on
  `SearchState.form_fill`, persisted per A2A context.
- **`adapters/a2a.py`** — the executor is a driver of the model: one turn at a time per context, a
  human-in-the-loop response mapped to the text the skill reads before the run, and after the run a
  form reply (from `state.form_reply`) delivered as a completed task or — when the caller activated
  the extension — an `input-required` pause carrying the HITL payload on the same task.
- **`adapters/a2a_hitl.py`** — the form-fill side of an A2A turn (`FormTurn.begin` / `finish`): which
  session the skill continues, the human's response as contract text (an approval becomes `yes` / `no` /
  a JSON object of corrections), and the reply as a `FormStop` (contract text + the pause payload, bound to
  its task). The executor edits no form state itself.
- **a2a-sdk 1.x** — the A2A endpoint speaks protocol v1.0 only (protobuf types, `SendMessage`, the
  `A2A-Version` header, a new task announced before its first status). kagent 1.x's remote tool is a v1
  client, so this is what makes the HITL work reachable at all. The CrewAI A2A demo was removed: CrewAI
  pins a2a-sdk 0.3 and cannot speak v1.
- **`agent.py`** — the handoff toolset next to core's MCP toolset. **`app.py`** — builds the skill and a
  core toolset for it.
- **`capabilities/hooks.py`** — `WriteToolGate` hides `create_workflow` / `resume_workflow_process` /
  `abort_workflow_process` from the model.
- **`tool_names.py`** — the workflow tools, their parameter names, `WRITE_TOOL_NAMES`, and the local
  `START_WORKFLOW_FORM_TOOL`.
- **`capabilities/plugins/workflow.md`** — advertises the skill on the A2A card; tells the model a request
  to create/modify/terminate is a workflow start, not a search, and to hand off (owns `list_workflows` and
  the handoff tool).
- **Tests** — `tests/test_form_fill.py` (contract, skill, handoff, literal contract, structured replies,
  the review regressions), `tests/test_form_prefill.py` (engine), `tests/test_kagent_hitl.py` (wire
  shapes), `tests/test_form_capability.py` (the capability on a real pydantic-ai run: claim, handoff,
  unknown key, failure), `tests/test_adapters.py` (executor: form replies, HITL pauses and resumes),
  `tests/test_capabilities.py` (`WriteToolGate`), `tests/test_plugin_loader.py`, `tests/test_state.py`.

---

## Tasks

- [x] Contract: fields, parser, renderers.
- [x] Skill: walk from page 0 each turn, stop on missing required, ask optional-only pages once, confirm,
      corrections re-walk, cancel, rejected pages relayed, subscription narrowing with reasons.
- [x] Model handoff: `start_workflow_form` tool, the first walk in the same turn, misremembered keys
      offered as a choice.
- [x] kagent HITL: extension on the card, pauses and resumes on one task, header alias.
- [x] Wiring: the skill as a pydantic-ai capability (`SkipModelRequest`, capability-owned tool); A2A
      executor + app; `WriteToolGate`; plugin text.
- [x] Tests, ruff, mypy; live e2e on the demo app and a production-shaped clone; three-reviewer final review.
- [x] Jev as the optional engine (stacked branch): routing in front of the model, option prefill,
      mid-form and confirmation reading.

## Live e2e (minimal core app on `main`: one product, `create_demo_lightpath`, dynamic second page)

Core on `main` with `MCP_ENABLED=true` (scratch app: product "Demo Lightpath", `create_demo_lightpath`;
page 1 = customer_name / speed / speed_policer, page 2 = redundancy + ticket_id when speed ≥ 10 Gbit/s,
else ticket_id only), the agent on A2A (`JEV_API_KEY` set, `gpt-4o` for the non-form path), one
`contextId`, the driver script acting as the calling agent:

| # | Caller sends | Skill replies | Decisions (agent log) |
|---|---|---|---|
| 1 | "create a demo lightpath for Universiteit Twente at 10 gig, policer on. Ticket JIRA-4821." | page 1: needs `customer_name`; filled: product, speed 10000, speed_policer true | routing intent 1.0 / workflow 1.0; product 0.86; speed 1.0; policer 1.0 |
| 2 | `customer_name: Universiteit Twente` + `ticket_id: JIRA-4821` | page 2: needs `redundancy` (options listed); ticket_id already taken | redundancy → not specified |
| 3 | `redundancy: protected` | summary of all six values, asks to confirm | — |
| 4 | `yes` | "Started workflow … Process id …" | approve 1.0 |
| 5 | "how many workflows are there?" | model path (aggregate + artifact) | routing intent other 1.0 |

Core ran the process to `completed`; its recorded state matched every value; `created_by` is SYSTEM
(no invented reporter). An earlier run also exercised two corrections while confirming (`speed: 1000`
dropped the redundancy page from the summary; `speed: 10000` brought the remembered answer back).
No LLM call happened anywhere in turns 1–4.

Two things the live run taught, both now in the code:
- Jev decided `product` at 0.86 on the first walk and *not specified* on the re-walk once this agent's
  own templated reply was in the conversation state. Fix: only the caller's turns are Jev's state, and a
  decided field is stored as an answer (no re-deciding; corrections arrive as values in a JSON reply).
- A bare "yes" scored 0.79 on the approval `Noul` with the value list in the state — under the 0.8 gate.
  Fix: a three-way `Choice` (approve / cancel / other, with a real confidence) over a minimal state
  ("question asked" + reply): 1.0 on the same "yes".

Known limitation: an active session claims every message on the context until it is done or cancelled;
an unrelated question mid-form is answered with the pending form message (the caller sees a form is in
progress and can relay that). Cancelling works (`stop` / `cancel` / `never mind` → Jev `Choice`).

## Live e2e on production-shaped workflows (database clone)

A clone of a production database (hundreds of subscriptions, about ninety user-facing workflows, tens of
thousands of processes) with a scratch app on core `main` registering five of its workflow names, forms
modelled on the inputs its processes were started with (`process_steps` of the latest completed run of
each): a create-lightpath workflow (product → customer → a list of exactly two service ports from the
customer's real ports / speed / bools / contacts / ticket), a create-service-port workflow (nine products of
two families → customer → port / mode / admin state / bools / native vlan), two terminate workflows and
`modify_note` (core's own `ModifySubscriptionPage` first: a raw subscription UUID). Driven over A2A
by a script standing in for the calling agent.

| Flow | Outcome |
|---|---|
| Create lightpath for customer A, 1 Gbit/s, vlan 10, policer on | Routing 0.79 over 90 keys; single-product page taken silently; customer A 0.97; speed + policer 1.0; `remote_port_shutdown` left to its default; asked only for the two service ports (listed by name) as JSON; ticket sent early and applied on page 2; process `completed`, state exact. |
| 10G service port for customer B, tagged, LLDP | The whole form from one sentence: product 1.0, customer B 0.98, IMS port 0.99, tagged, LLDP — zero questions, straight to the summary; ticket added as a correction; process `completed`. |
| Same without the product family | the other family's 10G on top at 0.56 → abstained and offered the nine products (correct: genuinely ambiguous). |
| Terminate customer A's lightpath | Core: `subscription.not_in_sync`. Reported as "the terminate-lightpath workflow — cannot run on this subscription now: subscription.not_in_sync"; naming the key does not force it. |
| Terminate customer C's lightpath | In sync itself, relations not: `subscription.relations_not_in_sync`, reported the same way. (No lightpath in the dump is terminable.) |
| Terminate customer D's IP subscription | Routing 0.95, confirmed runnable by core, subscription page filled from the bare UUID, defaults shown, ticket added, process `completed`. |
| Modify the note of one of customer A's subscriptions | Routing 0.99; bare UUID filled the subscription page; note sent structured; process `completed` with the note. |

What the production shapes forced into the code (all tested):
- **Structured fields** (`ListOfTwo[ServicePort]`, `contact_persons`): a `json` kind whose ask spells out the
  shape (count, keys, allowed values with labels) and whose value is JSON.
- **Subscription pages**: a `uuid` hint pointing at the search skill; the subscription id the model passes
  with the handoff fills the page (the earlier scan for a bare UUID in the message is gone: a heuristic),
  and the named subscription is checked through core's `get_subscription_available_workflows` — including
  *why* the intended one cannot run.
- **Routing over ~90 real keys**: its own lower gate (0.6; recoverable), an abstain that offers the
  likeliest candidates from Jev's distribution, a `choosing` state that keeps the question open, and
  candidates persisted as a list (JSONB reorders object keys).
- **Jev wording**: "which value did the user give" made Jev decline a customer named by label (0.61);
  "which option matches what the user asked for, by label or synonym" gives 0.99. Single-option choices
  are taken without asking anyone. Decisions stick as answers; only the caller's turns are Jev's state.
- **Summaries** list the defaults that will apply and render structured values as JSON; core's
  validation errors are relayed as their field messages, not the raw error dict.
- Operational: the agent fails to start when core is unreachable at boot — fastmcp's connection error
  is not among the types `verify_tool_contract` tolerates (pre-existing; the tolerance list needs it).

## kagent human-in-the-loop: native prompts from a remote agent

Verified from kagent source (`go/adk/pkg/tools/remote_a2a_tool.go`, `go/api/a2a/hitl.go`, the Python
runtime's `_remote_a2a_tool.py`/`_hitl.py`) and release notes: since **1.0.0-alpha1 (2026-09-18)** kagent's
agent-as-tool pauses the parent (`ctx.RequestConfirmation`) when our task ends in `input-required` with a
valid extension payload under `https://kagent.dev/extensions/hitl/v1`, shows the human our questions or an
Approve/Reject for our call, and resumes the *same task* with the response; a second pause loops. Without
the payload the parent errors ("requested input without a valid HITL extension"). v0.10.x used a legacy
variant and did not activate the extension outbound — this targets ≥ 1.0.0-alpha1.

Implemented:
- `state.py` — the skill returns `Reply(text, ask | approval)`: the same stop as data.
- `adapters/kagent_hitl.py` — wire models mirroring kagent's json tags; `ask_request` / `approval_request` out,
  `parse_response` / `answers_as_text` / `approval_decision` in (answers become the JSON-object contract,
  so the skill logic is shared); `PendingAsk` remembered on the session.
- `adapters/a2a.py` — card declares the extension; activation is echoed; a stop → `requires_input(final)`
  with the payload and `message.extensions`; on resume the response is mapped and the skill continues, or
  `resolve_approval` runs; the thread is keyed on the **task id** in this mode (the tool's shared
  `contextId` is process-wide in kagent's default "Shared" isolation, so task-scoped state is the correct
  unit). The activation header is `A2A-Extensions`, which a2a-sdk 1.x reads natively (superseded: an alias
  middleware was needed on 0.3).
- A page with only optional fields that nobody said anything about is asked once (a modify-note page *is*
  its optional note); an empty answer keeps the default.
- Tests: payload shapes, parsing, mapping; executor pause / resume / approval / no-extension.

Verified without a kagent cluster, two ways:
1. **Conformance** — kagent 1.0's own `kagent-core` HITL module (GitHub `main`, run with its a2a-sdk 1.x)
   parses our wire JSON (`ask_user_request`, `tool_approval_request`), validates request/response pairs,
   and the responses it builds are understood by our parser (scratchpad `e2e/conformance_*.py`).
2. **Live** — `hitl_drive.py` behaves like `remote_a2a_tool` (activates the extension, shows the pauses,
   resumes the same task with the human's answers). Against the production-shaped app: create lightpath — one
   task, paused for the ports/optionals, then for approval; empty answers kept defaults; approved →
   process `completed`, state exact. Service port: straight to approval; a rejection carrying
   `port_mode: untagged` + a ticket re-walked and re-asked approval with the corrected values; approved →
   `completed`. Modify note: a plain rejection cancelled.

Not verified: kagent's UI rendering of a *nested* remote's questions (inferred from `BuildHITLStatusMessage`,
which copies our questions to the top level), and whether the UI lets a human leave an optional question
empty (kagent validates only one answer per question; our mapping treats an empty answer as "keep the
default"). `InMemoryTaskStore` means a paused task does not survive an agent restart.

## Final review (2026-09-24)

A read-through of the branch as a reviewer, with the fixes it produced (each with a test):

- **Native choice answers looped.** HITL questions offered `value (Label)` strings that the option matcher
  did not accept, so a human picking a choice would have been asked again forever — the live runs had only
  exercised JSON and free-text answers. Choices are now the labels, and the matcher also accepts the
  `value (label)` rendering. Verified live: a product picked by label from nine choices.
- **Ask-once could loop.** An unparseable reply to an optional-only page re-asked it indefinitely; the
  page's fields are now marked asked, so any reply moves on with the defaults.
- **A confident but inapplicable pick walked into a 400.** A workflow the named subscription does not offer
  at all is now reported as "not offered for this subscription"; and the choosing step no longer drops
  earlier reasons or auto-picks the sole runnable workflow the user never asked for.
- Malformed HITL metadata is ignored instead of failing the task; an unmatched answer is logged.
- Jev's state is capped to the caller's last 20 messages; summaries list defaults only for the pages of
  the last walk; the unused message-history conversation builder was removed.
- A positional test edit had silently dropped three skill tests; restored.

A second, independent review pass (code-review at high effort) found ten more issues; all fixed, each with a test,
and the terminate flow re-verified live over A2A afterwards:

- **Memory keyed on the task, not the context.** In HITL mode the thread id was the task id, so every new task
  of a conversation started with empty memory. The thread is the A2A context again; the form session is tagged
  with its task id and a session left by another task is not continued.
- **A skill failure failed the whole task.** Jev or core being unreachable during the form step now falls
  through to the model instead of erroring the task.
- **Multi-select HITL answers were joined with commas** and a label containing a comma split in two; they
  travel as a JSON list now.
- **A message that is not about the form** (a question, a search) was swallowed as "continue" and the form
  re-asked; Jev now judges cancel / unrelated / continue. Unrelated messages go to the model and the form
  waits; while a workflow is still being chosen, an unrelated message drops the empty session.
- A read-only single-option field was auto-submitted; a bare UUID could overwrite a subscription id given
  explicitly; the workflow catalogue was fetched on every turn (now cached 5 minutes); the conversation
  and the page walk are capped (40 turns, 50 pages) so a form that never completes is rejected, not looped.

## Live re-verification with the model handoff (2026-09-24, base branch, no Jev)

The production-shaped flows above were run again over A2A with routing done by the model's
`start_workflow_form` handoff and the literal contract after it: create lightpath for customer A (customer by
label, the two service ports as JSON, speed, policer, ticket → started), 10G service port for customer B
(nine products offered, product and port picked by label → started; also as kagent HITL pauses with every
choice picked by label → started), terminate customer D's IP subscription (bare UUID → started),
modify the note of one of customer A's subscriptions (text and HITL → started), terminate the out-of-sync lightpath of customer A (blocked
reason reported, with what the subscription can run instead). What that run changed:

- The handoff tool takes the subscription id the model knows from earlier in the conversation, so the
  availability check runs even when the opening message does not repeat the id.
- A create workflow is not vetoed by a subscription id mentioned alongside (the model passed an existing
  subscription of the customer to a *create* handoff, which used to block it).
- When the intended workflow is blocked, the choice also lists the runnable workflows core offers for that
  subscription; when nothing is runnable the form closes with the reason instead of asking forever (the
  HITL variant of that loop had no way out).
- The plugin tells the model to hand off as soon as the target and the kind of thing are clear; product
  variants, speeds and ports are the form's questions, not the model's.
- Every contract word (`yes` / `no` / `cancel`, `true` / `false`, `ACCEPTED`), core's `subscription_id`
  name and the target enum are defined once (`render.py`, core's `Target`) and used everywhere.

## Final review of the two branches (2026-09-24, three independent reviewers)

Three review passes (simplification, hardcoding and Python standards, correctness with reproduced
scenarios) produced 40-odd findings; the ones that held up were applied together and re-verified live:

- **Correctness.** A JSON object inside a `field: value` line was applied as top-level answers (it could
  silently swap the subscription id); create workflows were vetoed or auto-converted to a modify by
  subscription narrowing in the unsure paths; a failure mid-walk left a half-reset session that a later
  `yes` would have submitted; the subscription the model passed on was lost when the text held another id;
  a stored choice was submitted against regenerated options; `Accept` consent was carried across pages;
  a rejection whose correction did not parse cancelled the form; quoted or punctuated `yes` was not
  literal; a paused HITL form was wiped by a message on another task; a start that timed out could be
  retried into a duplicate process; two messages for one context raced. All fixed, each with a test.
- **Simplification.** One first-turn path for asker routing and the model handoff (the routing turn no
  longer asks the engine and core twice); caller-only session turns; the walk commits pages only when it
  stops; the prefill plans are built from the same `FormField`s the contract uses (one schema classifier,
  no contract → prefill import); one need-input builder; one delivery path in the executor for claimed
  turns and handoffs; the write-only `complete` flag and dead result fields removed.
- **Hardcoding and standards.** Every contract word, core's `subscription_id` name, the form tools'
  parameter names and the target enum live in one place; `FieldKind` is a `Literal`; `ModelRetry` is the
  only exception read as "core refused"; typed `SearchState`/`FormFillSession` instead of `Any`; a
  discriminated union parses HITL responses; the create call the human approves is the one sent.
- **Model handoff hardening (found live).** A misremembered key becomes a choice among the real keys it
  resembles instead of a silent drop, and the plugin tells the model not to search for a product or
  subscription first — with that, the plain-language service-port request handed off three times out of three.

## One structured contract for every caller (2026-09-24)

The caller contract after the handoff is now **a JSON object keyed by field name** for every caller, kagent
or not; the `field: value` line grammar (and the regexes that parsed bullets, backticks and `=`) is gone.
Research first: the A2A v1.0 spec standardises only the `input-required` pause, no question or form
schema and no extension for it (the samples' `{"type": "form"}` DataPart is a sample convention; the
maintainers' schema-contract extension is a proposal). kagent's `ask_user_response` is positional (answers
in question order, no field names) and cannot carry values for pages not yet asked, so it stays the kagent
transport envelope and is mapped onto the same JSON object (`answers_as_text`), not adopted as the
universal contract. The JSON stays in the text part because the known callers text-extract (follow-up 4).

- A reply is the object and nothing else (no code fence, no prose around it); prose is never mined for values, and
  an object embedded in prose is not the contract — a nested object is its field's value, so a caller can
  no longer swap the subscription id by quoting one inside another value.
- Values are not judged by the agent: they go to core as sent, and core's pydantic-forms validation is the only
  validation (its per-field messages are relayed). An agent caller reads the stop and sends the format it asks
  for; through kagent the answer is a person's, forwarded verbatim: a picked chip is a label the adapter maps
  back to its value (`PendingAsk.options`), and free text core rejects is interpreted **once, a page at a time**
  into the values the fields expect (`form_fill/interpret.py`: the `Interpreter` protocol takes the page's
  rejected fields with the person's words and returns the values; `ModelInterpreter` = one pydantic-ai run with
  an output model typed per field, `SystemOneInterpreter` = a typed decision engine such as Jev over the
  page-prefill plans and gate — the swap is the one `FormFillSkill.interpret` attribute), then core decides
  again. The same words
  are never interpreted twice (`FormFillSession.interpreted`). A rejection is re-asked as the rejected fields
  themselves (chips, with core's reason in the question), never as "send JSON" — a person never answers in
  JSON. Verified live: "vlan twenty" → 20 and a name, an e-mail address and a phone number in one sentence → the
  contact-person object, after core rejected the strings; labels sent by an agent → the UUIDs.
- Keys are the field names exactly; values may be JSON-typed (numbers, booleans,
  lists for multi-selects) or the strings the stop showed (labels, `value (label)`).
- No token parsing at all (2026-09-27): a decision (start, cancel) is kagent's structured approval or the
  interpreter's reading of the message, never a word matched by this code — "yes, but change the speed" must not
  start anything; a correction at the
  summary is a JSON object; a kagent rejection whose reason is a JSON object is a correction.
- Choosing a workflow: the key alone, or a JSON object naming it and carrying values.
- `render.py` keeps one regex, the UUID finder for a bare subscription id in the opening request.
- Core's tool results are validated with core's own response models (`WorkflowFormPage`, `WorkflowSchema`,
  `SubscriptionWorkflowListsSchema`, `ProcessIdSchema`) and its tool arguments are built from core's request
  models (`GetWorkflowFormRequest`, `ListWorkflowsRequest`, `SubscriptionIdRequest`); the validation-error
  body fastmcp relays as text is validated as pydantic-forms' own error shape. The agent depends on
  orchestrator-core, so nothing is mirrored. What remains hand-parsed is the browser-oriented page schema
  (follow-up 5); pydantic-forms' field types cannot be imported here because its validators package needs
  `email-validator`.

## One personality per branch (2026-09-24)

The engine-driven paths — routing in front of the model, the `choosing` state with candidates and blocked
reasons, page prefill from the conversation, the asker's mid-form and confirmation judgements, `prefill.py`
with the `SystemOne` protocol — only ever ran with Jev, so they are gone from this branch (user: "if it's
only used by Jev, why not remove it in this branch"; "make it swappable if possible but don't write code just
for Jev"). The Jev branch owns them. What replaced them here:

- The handoff tool checks the key against core's catalogue itself and raises a tool retry on an unknown one,
  so the model corrects its own pick (no close-match guessing, no choice offered to the caller).
- A handed-off workflow core cannot run on the named subscription closes with core's reason and the
  workflows the subscription can run now (`render_blocked`); the caller restates its request.
- `FormFillSkill` is the walk only (`interpret` is its one engine seam); `FormFillSession` lost
  `candidates` / `blocked` and the `choosing` status.

## Six small cuts (2026-09-24, after the review)

One `values` dict on the session instead of `answers` + `pending` (values are applied to a page when its
field appears; nothing is coerced); no UUID scanning at all — the subscription id is what the model passes
with the handoff, and the form's own subscription page asks for it otherwise (the last regex is gone); `request: str` instead of the engine's `turns` list;
one handoff function (the checked tool); `fields` is reset per walk so it *is* the last walk's field set
and `walked` is gone; the reply dataclasses live in `state.py` next to the session and `form_reply` is typed.

## No word is parsed (2026-09-27)

The last literal contract went: no `yes` / `no` / `cancel` tokens, no `FormCommand`. A message on an open form
is either the data model (a JSON object of values, one pydantic parse) or it is read by the `Interpreter`
against the current stop — `Interpreter.message(fields, text, decisions)` returns values and a decision, the
decision only when the message states it outright and on its own (a start next to changed values is not a
start: the values are walked and the new summary asks again). kagent's approval arrives as a structured
`Decision` on the state (`FormTurn.install`), a JSON rejection reason as data, any other rejection as cancel.
The human's answers travel the same way (`form_values` on the state, 2026-09-28): a chip is mapped to its
value by lookup, free text stays as typed, nothing is serialised to JSON text and parsed back. Structure is
mapped, words are interpreted.
The stops now say what a reply may do in words or as data instead of quoting tokens. What remains in
`form_fill/` is data models (`Reply`, `Interpretation`, `FormFillSession`), the walk, the interpreter, the
stop renderers, and the schema bridge whose schema-reading half goes with core follow-up 5.

## One pydantic model per form page (2026-09-28)

The hand-written `FormField` records (a `FieldKind` per field, a `shape` string for structured fields, a
`value_type` mapping for the interpreter, prose describers for the stops) are gone. `core_bridge.page_model`
builds **one pydantic model per page** from core's page schema with `create_model` — display-only fields
left out; an enum as a `Literal` of its values with the labels kept as field metadata
(`json_schema_extra["labels"]`); `Accept` as `Literal["ACCEPTED"]`; an array of an enum as `list[Literal]`
with its item bounds (a `maxItems: 1` single-select stays a one-element list, as the form wants it); an
object or a list of objects as nested models built from `$defs`; required, default, description and the
form's `format` carried — pure and cached per schema. That one artifact drives everything:

- the interpreter's output type is its partial variant (every field optional, plus the decision literal;
  `interpret.reading_type`), with the fields' schema in the prompt, so a person's words become typed nested
  values without a per-kind mapping;
- the stop for an agent caller is core's verdict on the page plus the page model's JSON schema (`Rejected by
  the orchestrator: …` in core's words, `Page schema: {…}`, `Filled so far`, how to answer) instead of prose
  descriptions;
- the kagent questions are one per model field, chips from its allowed values shown by label (a list field's
  answer is a list, so `multiple` is set for it); nothing spells out what a field expects — a person answers
  in words, the interpreter has the schema, and core's validation message comes back on the re-asked question;
- the summary shows values with the labels the model carries.

`FormFillSession.fields` became `pages`: each walked page's schema as core sent it, from which the models
are rebuilt each turn (a dynamic model cannot be persisted); readings that span the form (a correction at
the summary, a rejected field's lookup) use `form_model(session.pages)`. Nothing about the contract changed
for the caller except the shape of the need-input stop; kagent's wire and the executor are untouched.

**Core is the only judge of a page (same day).** The walk no longer decides locally which required fields
are missing: every page is submitted to core as it is known, and core's 400 — `Field required` per missing
field, its message per wrong value — is the stop. `need_input` and `render_rejected` merged into one
`render_stop` (core's verdict, the page schema, filled so far, how to answer; as questions: the rejected
fields with core's reason, then the untouched ones), and the skill lost its missing-fields branch and
`_rejected`. A caller sees every problem of a page in one turn (a label sent as a value used to surface
only after the missing field was supplied). One local rule stays: a page of optional fields nobody has
touched is asked once, because core would accept it as it is. Cost: one extra `get_workflow_form` call per
stop. The hand-written "what this field expects" wording of a person's question went with it: a question
is the field's title, whether it is required and its default, chips for its options, and core's message
when it did not accept what it got.

**Every reply is data (same day; user: "feed everything directly back to the model from core").** `render.py`
is gone. Every answer of the skill is one `FormReply` (`state.py`), serialised as the JSON text of the
reply: `status` gathering (`page`, `title`, the page model's `schema`, core's `rejected` errors as
pydantic-forms reports them — `form_errors` validates the relayed 400 body as `ErrorDict`s, no message
flattening — or a `reason` when core gave no field errors, the `values` known so far), confirming (`values`
to be submitted, `defaults` that apply), started (`process_id`), cancelled, failed (`reason`). The agent card states that shape once, in place of the how-to-answer paragraph every
stop used to repeat. What is not core's data and still has to be built here: the kagent questions and chips
(`skill.questions` / `question`, from the page model) and the approval envelope for `create_workflow` (its
hint is one line; the values are the call's `args`). A start core refuses reopens the form (`gathering`
with the form's schema and core's errors); a form that never completes fails and closes.

**No pre-check either (same day; user: "can the same be said for `_blocked`?").** The per-subscription
availability check at the handoff (`get_subscription_available_workflows`, `Target` tracking of create
workflows, the `blocked` reply with runnable alternatives) is gone: the subscription the model passes
fills the form's first page, and core's own subscription-page validator rejects one the workflow cannot
run on, with its reason — that rejection is the reply like any other, and the caller may correct the id
or cancel. A start that fails without core's answer closes the form with the error text as its `reason`.
By the same argument (user: "find any other similar functions that can be removed"): `open` no longer
re-checks the handed-off key against the catalogue — core's refusal on the first fetch closes the form as
`failed` with core's text; `_see_page` no longer treats an empty string as "not given" — a value goes to
core exactly as sent, and the kagent adapter leaves an unanswered question out of the values instead of
sending `""` (an all-empty answer still reaches the skill as `{}`, so the page's defaults apply; words for a
free question go to the interpreter).

## Out of scope / follow-ups

0. **Persist A2A tasks** (own PR, before HITL is relied on in production or the deployment goes
   multi-replica): a2a-sdk 1.x's `DatabaseTaskStore` (`postgresql` extra, `a2a-db` migration) so a paused
   human-in-the-loop task survives a restart and can resume on another replica; the per-context lock then
   needs a database-level equivalent. Changes runtime behaviour and deployment, hence not in this branch.
1. **Eval on real WFO forms** — Jev's pick accuracy per field kind and the routing false-positive rate,
   to tune `JEV_PREFILL_THRESHOLD` (one threshold today). The confirmation summary is the safety net.
2. **Suspended processes** (`get_process_status().form` → `resume_workflow_process`): the same skill
   shape, one page at a time, each resume commits.
3. **Large pickers** (thousands of subscriptions): the contract points the caller at the search skill
   for an id; a Jev `Choice` cannot hold them.
4. **Structured A2A parts**: the contract is text because ADK's `AgentTool` text-extracts; a `DataPart`
   twin of the need-input message can be added when the parent side consumes it.
5. **An agent-facing form page in core** (core PR, then a version bump here): `get_workflow_form` returns
   the raw pydantic-forms JSON schema, written for a browser (`$ref` / `allOf` / nullable `anyOf`,
   `uniforms` widget hints, display-only `format` markers, enum labels in a side table, nested `$defs`
   for structured fields) and validation errors as a Python-repr'd dict inside the MCP error text. Core
   has the lossless source (the form model) and could return a flat per-page field list — name, title,
   kind, required, display-only, default, options as value/label pairs, shape of structured fields — and
   errors as a list of field and message. Everything that undoes the browser schema and the error text on
   the agent side is isolated in **`form_fill/core_bridge.py`**: its schema-reading half is deleted and
   `page_model` is built from core's field spec instead; the built model, and everything that works from it
   (the interpreter's output type, the stops, the questions, the summary), stays as it is.
