# Hybrid workflow form-fill — Implementation Plan

**Date:** 2026-11-08
**Decision:** the deterministic walk (chips, interpreter, widget resolution) is replaced by a hybrid: the
state machine, the approval and every option list stay in code; the model fills each page in
conversation with the person. Recorded on 2026-10-22/23 (research session, memory
`per-page-llm-form-fill-research`) and decided by the user on 2026-11-08: "more conversational and the
model will fill everything in, that way we don't need custom components for each form field".
**Branches:** `feat/hybrid-form-fill` off `main` in orchestrator-agent and in orchestrator-agent-chat.
The export branches (`export-download-route`, `export-download`) merge on their own; the overlap is a
few lines in `adapters/ag_ui/contract.py` and `worker.py`.

## What stays in code (non-negotiable)

1. **The machine.** Accepted pages (`page_inputs`), the current page as core described it, `submit_page`
   with the current page's JSON Schema as its parameters, `restart_from_page`, `cancel_form`. Core is the
   only validator (`get_workflow_form` with `verdict="result"`).
2. **The start.** When core says the form is complete, code raises the approval: the card is rendered from
   the accepted pages with labels (`FormReply` `confirming` + `Approval`), the person decides through the
   client's native approval (AG-UI `tool_call` interrupt, LibreChat ask-user card, kagent
   `tool_approval_request`). The model never has `create_workflow` and never writes the summary. On a route
   that cannot return a structured approval the form is filled and ends `prepared`, nothing starts.
3. **Options.** A field with a short list is in the tool schema (`enum` + `options`, as core or the
   options-source reader writes it); a long list is reachable only through `search_options`, and a value
   for such a field must be one a search in this session returned (provenance; no invented values).
4. **Obvious values.** Code fills what needs no one: an `accept` field (the approval at the end is the
   person's consent, before anything is written) and a required choice with exactly one option.
5. **Guards.** A tool-call budget per page (then the tools say: ask the person), one start per approval,
   idempotent submits (core re-validates), a start core refuses reopens the page it names.

## What the model does

Reads the person's words into the current page, searches, asks in plain text for what is missing (one
message per page, options named when there are a few), fixes what core rejected, acts on a rejection
reason, corrects earlier pages with `restart_from_page`, ends the form with `cancel_form` when the person
moves on. The instructions are rendered per request (workflow, accepted values with labels, the current
page's fields, core's last verdict, searches so far, the rejection reason). Model floor: gpt-4o.

## Agent (orchestrator-agent)

`form_fill/`:
- `state.py`: `FormInput(decision, reason)`; `FormFillSession(workflow_key, status gathering|confirming|done,
  pages, page_inputs, draft, rejected, found, calls, pending, unseen, reason)`; `FormReply` gains `prepared`;
  `AskField` is gone.
- `pages.py` (new, replaces `walk.py`): `page_of` (core), `obvious`, `tool_schema` (the page as the
  parameters of `submit_page`), `describe` (a page as a tool result tells it), `labels`.
- `tools.py` (new): `PageToolset(AbstractToolset)` — `submit_page`, `search_options`, `restart_from_page`,
  `cancel_form`, built per run step from the session.
- `instructions.py` (new): the per-request instructions while a form is open.
- `handoff.py`: `start_workflow_form` fetches page 0, submits a passed subscription id by code, answers the
  model with the page to fill.
- `skill.py`: `FormFillSkill` — the catalogue, `open`, `submit`, `search`, `restart`, `cancel`, `summary`,
  `start`, `handle` (the approval decision).
- `capability.py`: `get_toolset` (handoff + pages), `get_instructions` (dynamic), `before_run` (the person's
  decision, before anything is rendered for the model), `before_model_request` (the decision's outcome, and
  the summary as soon as the form is complete, in the model's place).
- `session.py`: the writes and projections, smaller. `pending.py`: the approval stop only.
- `widgets/`: `base.py`, `source.py`, `enrich.py` stay (the options-source reader); `resolve.py`,
  `waiting.py` go. `interpret.py`, `questions.py`, `verdicts.py`, `walk.py` go.

Transports: every ask path goes; the approval stays. `adapters/ag_ui/interrupts.py` (`tool_call` interrupt,
`ApprovalAnswer(approved, reason)`), `adapters/chat/librechat.py` (approval card; typed text under it is a
rejection with that reason), `adapters/a2a/kagent.py` (`tool_approval_request`, `rejection_reason`).
`GET /chats/{thread}/options/{field}` goes. `adapters/words.py` words the approval and the outcomes.
`capabilities/plugins/workflow.md` tells the model to hand off and then fill. README follows.

Tests: `tests/test_form_fill.py` (the toolset over the fake core: accepted, rejected, complete, provenance,
budget, restart, cancel, obvious values, summary labels, start, refused start), `test_form_capability.py`
(a scripted `FunctionModel` calling the tools; the summary skips the model; decisions), `test_form_session.py`,
the transport tests (approval only), `test_agui_chats.py` minus the options route, `test_openapi.py`.

## Popup (orchestrator-agent-chat)

`StopCard` becomes `ApprovalCard` (Approve and start / Reject); `Combobox` goes; the `searchOptions` /
`options` hooks and the `AskStop`, `AskedQuestion`, `AskAnswers`, `FieldOption` types go. Typing while the
approval card is open sends the message *and* resolves the interrupt as rejected with the text as reason, so
"make it 1G" is a correction, not a cancel. Empty-state copy and README follow. Contract regenerated from the
agent (`npm run contract` after `uv run python -m orchestrator_agent.adapters.ag_ui.contract ...`).

## Verify

`uv run --no-sync ruff check`, `mypy`, `pytest`; `npm run check` in the popup; then the live stack (core on
8080, agent on 8085 restarted on this branch, popup on 5173): "Create a 10G service port for <customer> on
<node>, tagged" fills the pages in conversation and ends on the approval card; a typed correction under the
card reopens the page; approve starts the process on the clone.

## Later (not in this PR)

- A2A data parts with field values submitted by code (data first, model second) for agent callers.
- A structured `ask_user` for agent callers over kagent's extension, when a parent model must not relay.
- Form cases in `evals/` (valid payloads, questions asked, invented options at zero).
