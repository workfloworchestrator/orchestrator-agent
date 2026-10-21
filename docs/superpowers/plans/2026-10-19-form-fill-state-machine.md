# Form-fill as an explicit state machine

> Status: in progress on `feat/form-fill-state-machine` (steps 1 to 4 done 2026-10-20). A refactor with no behaviour
> change, on top of the widgets branch. Every step leaves the suite green; the suite is the safety net (445
> tests before step 1, 478 after step 2).

## Why

`form_fill/skill.py` is 859 lines: a 408-line `FormFillSkill` with 22 methods and 29 module-level helpers.
The session's phases exist (`opening`, `gathering`, `confirming`, `done`) but the transitions between them
are implicit in whoever touches `FormFillSession`: it is written from the turn, the walk, resolution,
recording, stopping, seeing a page, reinterpreting and starting. Every new rule became another branch in
`_walk`, until the complexity limit forced a helper out. The helpers fall into four clusters that share no
state, kept flat behind underscores in one file.

The design itself is right and stays: a deterministic walk over core's pages with core as the only judge,
the model at the edges only (interpreter, chooser), a persisted pause per human stop, re-walking from page 0
each turn. What changes is how it is written down.

## The machine

### Phases

The four statuses stay as they are, since adapters, replies and persisted sessions use their strings. They
become one `FormPhase` enum (a `StrEnum`) instead of a `Literal`:

| Phase | Meaning | Pending stop |
|---|---|---|
| `opening` | handed off by the model, nothing walked yet | none |
| `gathering` | a page's questions are out | `PendingAsk(kind="ask")` |
| `confirming` | the start is out for approval | `PendingAsk(kind="approval")` |
| `done` | closed: started, cancelled or failed | none; the session is dropped |

### Events

One per turn, what the transport made of the human's response (today spread over `state.form_input`,
`state.hitl` and `session.unseen`):

| Event | Carries | Source (`session.event_of`) |
|---|---|---|
| `Opened` | the subscription the model passed | `open()` |
| `Answered` | field values | `FormInput.values`, when there are any |
| `Approved` | — | `FormInput.decision == START` |
| `Cancelled` | — | `FormInput.decision == CANCEL`; wins over answers |
| `Unanswered` | nothing: a message that is not a response | `form_input is None`, stop already shown |
| `Shown` | the stop is shown (again) | `session.unseen`, or a response with nothing to read |

### Transitions

Written as one table in one place and consulted by one dispatcher; every cell below is an existing path.

Actions: `WALK` (record any answers, walk from page 0), `START`, `CANCEL`, `ABANDON` (drop the session, the
message is the model's), `NOTHING` (not this form's turn). The same stop again is a `WALK`: the walk is a
function of the session, so it reproduces the stop.

| Phase | Event | Action | Next phase |
|---|---|---|---|
| `opening` | `Opened` | `WALK` | `gathering` / `confirming` / `done(failed)` |
| `opening` | anything else | `ABANDON` | session dropped |
| `gathering` | `Answered` | `WALK` | `gathering` / `confirming` / `done(failed)` |
| `gathering` | `Approved` | `WALK` (same stop) | `gathering` |
| `gathering` | `Cancelled` | `CANCEL` | `done(cancelled)` |
| `gathering` | `Unanswered` | `ABANDON` | session dropped |
| `gathering` | `Shown` | `WALK` (same stop) | `gathering` |
| `gathering` | `Opened` | `NOTHING` | `gathering` |
| `confirming` | `Approved` | `START` | `done(started)` / `gathering` (core rejects a page) / `done(failed)` |
| `confirming` | `Answered` | `WALK` (a correction) | `gathering` / `confirming` |
| `confirming` | `Cancelled` | `CANCEL` | `done(cancelled)` |
| `confirming` | `Unanswered` | `ABANDON` | session dropped |
| `confirming` | `Shown` | `WALK` (same stop) | `confirming` |
| `confirming` | `Opened` | `NOTHING` | `confirming` |
| `done` | anything | `NOTHING` | `done` |

Implemented in `form_fill/session.py` (`FormPhase`, the events, `event_of`, `Action`, `TRANSITIONS`,
`transition`) and consulted by `FormFillSkill._dispatch`; `tests/test_form_session.py` asserts every cell.

The walk is an action, not a phase: it consults core page by page and ends in exactly one of three
outcomes, a page stop, the summary, or a close with a reason. That is already true; the plan makes it the
walk's return type.

### The walk's page pipeline

Inside the walk, what happens to a page after core described it is an ordered list of checks, each a
function `(walk, page) -> Reply | None`, applied in sequence; the first that answers ends the turn:

1. `unanswerable` — no answer can satisfy the page (schema only): close with the reason
2. `miscounted` — a list answered with the wrong number of items: ask again with the count
3. `unresolved` — typed words no option fits: ask again with the candidates
4. `untouched` — a page of optional fields nobody touched, or a waiting field that has options now: ask once
5. `submit` — the page is known: send it and go on

New rules add a function to the list. `_decided_here` is the seed of this.

### Session writes in one place

`FormFillSession` is written only through named transitions in one module, and nowhere else:

| Transition | Writes | Called from |
|---|---|---|
| `opened(session)` | `status = gathering` | `_dispatch` on `Opened` |
| `answer(session, values)` | `values` | `_dispatch` on `Answered`; `_reinterpret` |
| `keep_resolved(session, name, value, label)` | `values`, `resolved` | `_record` |
| `drop_answer(session, name)` | `values`, `resolved` | `_record` |
| `choose_step(session, name, step, value)` | `values`, `steps` | `_record` |
| `step_back(session, name)` | `steps`, `values`, `resolved` | `_resolve` |
| `begin_walk(session)` / `saw_page(session, page)` | `pages` | `_walk` |
| `interpret_words(session, name, words)` | `interpreted` | `_reinterpret` |
| `stop_at(session, pages, phase, waiting)` | `page_inputs`, `status`, `waiting` | `_stop` |
| `mark_asked(session, names)` | `asked` | `_page_stop` |
| `consent(session, key, seen, name)` | `consents`, `values` | `_consented` |
| `paused(session, pending, unseen=)` | `pending`, `unseen` | the LibreChat and kagent adapters' `pause` |
| `close(session)` | `status = done` | `_dispatch`, `_walk`, `_decided_here`, `_page_stop`, `_start` |

Done in step 2. `tests/test_form_session.py` walks every module under `src/orchestrator_agent` with `ast` and fails on
any assignment to, or mutating call on, a session field outside `session.py`; it caught one write the manual
pass had missed.

## Files after

| File | Holds | Lines (target) |
|---|---|---|
| `form_fill/skill.py` | `FormFillSkill`: `handle`, `open`, `_dispatch` over the table, the summary, `_start` | 225 (≤ 300) |
| `form_fill/walk.py` | `Walk` (the collaborators), `walk`, `PAGE_CHECKS`, enrichment and resolution, reinterpretation, `page_stop`, `see_page`, consent, `labels`; `CallTool` | 419 (≤ 300: over, see below) |
| `form_fill/session.py` | `FormPhase`, the events, `event_of`, the transitions, the writes, the projections (`fields`, `given`, `fingerprint`) | 265 (≤ 200) |
| `form_fill/questions.py` | wording: `question`, `questions`, `unique_choices`, `offered`, the hint constants, `missing` | 208 (≤ 200) |
| `form_fill/verdicts.py` | schema-only decisions: `unanswerable`, `miscounted`, `count_of`, `single_pick`, `picked` | 107 (≤ 120) |
| `form_fill/widgets/steps.py` | cascade bookkeeping: `waits`, `waiting`, `goes_back`, `stepped`, `takes_only_option`, `CHANGE_STEP` | 64 (≤ 80) |
| `form_fill/pending.py`, `handoff.py`, `capability.py`, `interpret.py`, `model.py` | unchanged | |

`walk.py` came out above its target because the walk, the page's widget handling (enrich, resolve, record)
and what a page is submitted with (`see_page`, consent) are all there. The widget handling is the natural
next cut, into `widgets/walk.py` or similar, if the file keeps growing; it is not done in this refactor.

`FormFillSession` keeps its field names and status strings: persisted snapshots must still load (there is a
test for that already).

## Steps

Each step is one commit, behaviour-preserving, suite green.

1. **Phases and events.** Done. `FormPhase`, six event types, `event_of`, `transition(phase, event)` and the
   table in `session.py`; `handle` and `open` dispatch through it and `_turn` is gone. The table test covers
   every phase × event pair, and `event_of` its precedence (a cancel wins over answers).
2. **One writer.** Done. The transitions above in `session.py`; the skill and both adapters write the
   session only through them; the `ast` guard test is in.
3. **The walk out.** Done. `walk.py` holds the walk as functions over a frozen `Walk` (session, core's tools,
   widgets, graphql, choose, interpret); the branches are `PAGE_CHECKS = (unanswerable, miscounted,
   unresolved, untouched)`, applied in order, the first that answers ends the turn. The walk returns a
   `Reply` or `Completed`; the skill turns `Completed` into the summary.
4. **Helpers home.** Done, together with step 3 (the walk could not leave without them). Public names lost
   their underscore where they crossed modules (`count_of`, `picked`, `waits`, `missing`); two test imports
   moved to `questions.py`, one test calls `walk.see_page` instead of the skill's static method.
5. **Trim.** The module docstrings of `skill.py`, `walk.py` and `session.py` describe the machine. Left: the
   widgets plan and spec where they name `skill.py` internals.

## Not in scope

- A state-machine library (`transitions`, `python-statemachine`): four phases and a dict are enough, and a
  dependency buys nothing here.
- Changing the re-walk-from-page-0 model, the transports, the widget layer or any wording a person sees.
- The interactive form view (MCP Apps), the stop chip, the IMS port widget: separate work.

## Done when

- All existing tests pass with only import changes; the table test and the `ast` test are added.
- No file in `form_fill/` above its size target; `ruff` complexity stays under the limit without exemptions.
- `FormFillSession` is written in `session.py` only.
