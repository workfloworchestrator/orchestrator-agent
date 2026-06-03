# Fetch-entity-by-id fast path (`get_entity_by_id`)

**Date:** 2026-06-03
**Status:** Approved

## Problem

Users frequently reference a specific entity by its UUID — often a **partial UUID prefix** (e.g. the first 8 hex chars they copied) rather than the full id. Today, getting the full domain model for such an entity takes the long route:

1. Planner LLM call → emits a `search` task + a `result_actions` task.
2. Search skill run resolves the entity (lightweight id/title/score projection).
3. Result Actions skill run calls `fetch_entity_details(entity_id, entity_type)`.

Two problems:

- **Roundtrips:** for a request that is really "show me `<type> <id>`", we pay for a full search skill run purely to turn an id into a fetchable id.
- **Partial ids don't work at all** on the direct path: `fetch_entity_details` (`src/orchestrator_agent/tools/result_actions.py:39`) hits `GET /api/{entity}/{entity_id}` (`src/orchestrator_agent/settings.py:42`), a path param that requires the **exact** UUID. A prefix 404s.

## Goal

When the user gives an entity **type** plus a **full UUID or partial UUID prefix** and wants details, resolve it in a **single Result Actions skill run** (no separate search step), supporting partial prefixes, and returning the full domain model when the match is unambiguous.

## Decisions (from brainstorming)

- **ID forms in scope:** full UUID **and** partial UUID prefix.
- **Ambiguity behavior:** exactly 1 match → fetch full details; 2+ matches → return a candidate list (id + title) so the user can refine; 0 matches → friendly "not found".
- **Entity scope:** all four entity types (subscription, product, workflow, process), but the **entity type must be stated**. A bare id with no type word does **not** trigger the fast path — it stays on the normal search route. The tool requires `entity_type`; no type inference.
- **Approach:** a new `get_entity_by_id` tool on the Result Actions skill (not a deterministic pre-planner shortcut, not an overload of `fetch_entity_details`). The LLM planner decides routing; no code-level NL intent detection.
- **Resolution mechanism:** a **direct DB prefix query** against orchestrator-core tables (cheap, deterministic, no embeddings), not the semantic search engine.

## Design

### 1. Flow & cost

For "details for `<type> <id-or-prefix>`", the planner emits a **single `result_actions` task**. The Result Actions skill calls the new `get_entity_by_id` tool. Net effect: the separate search skill run is dropped, and partial-id resolution becomes possible. Cost becomes: planner LLM call + 1 Result Actions skill run (vs. planner + search run + result-actions run today).

The split of responsibilities is deliberate:

- **Resolution** (prefix → full UUID, or candidate list) is a **direct DB read** — it only needs `(id, title)`, which is a single cheap row lookup.
- **Detail fetch** (full UUID → full domain model) stays on the **HTTP API**, because the API assembles the full domain model (product blocks, instance values, etc.) that a flat DB row does not contain.

### 2. New module `src/orchestrator_agent/entity_lookup.py`

Owns the entity-type → table mapping and the prefix resolver. Kept separate from `result_actions.py` so the tool stays thin and the resolver is unit-testable without an LLM or HTTP.

Confirmed map (verified against orchestrator-core 5.0.2):

| EntityType | Table | id column | title column |
|---|---|---|---|
| `SUBSCRIPTION` | `SubscriptionTable` | `subscription_id` | `description` |
| `PRODUCT` | `ProductTable` | `product_id` | `name` |
| `WORKFLOW` | `WorkflowTable` | `workflow_id` | `name` |
| `PROCESS` | `ProcessTable` | `pid` | computed: `f"{workflow_id} ({last_status})"` (no name column) |

```python
@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    title: str

# Maps each EntityType to its table, id column, and how to derive a title.
# Title is a column name for SUBSCRIPTION/PRODUCT/WORKFLOW and a computed
# expression for PROCESS (which has no name column).
_ENTITY_LOOKUP: dict[EntityType, _LookupSpec]

MIN_PREFIX_LEN = 4

def resolve_entity_id_prefix(
    session: WrappedSession,
    entity_type: EntityType,
    prefix: str,
    limit: int,
) -> list[ResolvedEntity]:
    """Return entities whose id starts with `prefix` (case-insensitive), up to `limit`."""
```

Query shape: `select id_col, title_expr from table where cast(id_col, Text) ilike f"{prefix}%" limit (limit + 1)`. The `+1` lets the caller detect "more than `limit`" without a second count query. PROCESS has no title column, so its title is the computed `f"{workflow_id} ({last_status})"`.

### 3. New tool `get_entity_by_id(id_or_prefix, entity_type)` (in `result_actions.py`)

```python
@result_actions_toolset.tool
async def get_entity_by_id(
    ctx: RunContext[StateDeps[SearchState]],
    id_or_prefix: str,
    entity_type: EntityType,
) -> ToolReturn:
```

Behavior (dispatch via `match`/structural checks, no `isinstance` chains):

1. **Normalize & validate** `id_or_prefix`: strip; lower-case. Reject input containing non-`[0-9a-f-]` characters with `ModelRetry("'<x>' is not a UUID; search by name instead.")` — this signals the LLM to fall back to search.
2. **Full valid UUID** → skip resolution; fetch via `_fetch_entity_detail(...)` (section 4) → `DataArtifact`.
3. **Partial prefix:**
   - Shorter than `MIN_PREFIX_LEN` → `ModelRetry("Need at least 4 characters of the id to look it up.")`.
   - Call `resolve_entity_id_prefix(db.session, entity_type, prefix, limit=N)` (e.g. `N = 10`):
     - **1 match** → `_fetch_entity_detail(...)` for that UUID → `DataArtifact`.
     - **2…N matches** → return a **plain `ToolReturn`** (no `ToolArtifact` metadata) whose `return_value` is the candidate list (`[{"entity_id": ..., "title": ...}]`) plus a short instruction string; the skill LLM lists them and asks the user to refine the id or pick one. If more than `N` matched (we fetched `N+1`), include a "showing first N — refine the id" note.
     - **0 matches** → `ModelRetry(f"No {entity_type.value} found with id starting with {prefix}.")`.

**Why the candidate list carries no artifact:** `QueryArtifact` is a *reference* — the AG-UI adapter (`adapters/ag_ui.py:73-83`) emits only `metadata.model_dump_json()` and the frontend re-fetches full rows via `GET /queries/{query_id}/results`. A direct DB prefix resolution is **not** a persisted query, so it has no `query_id`; a `QueryArtifact` would dangle. Returning the candidates as plain tool content lets the skill LLM present them as text for disambiguation — no persistence, no frontend change. Tools that return no `ToolArtifact` fall through to the adapter's default handling, exactly like any non-artifact tool.

Each path records a `ToolStep` in `ctx.deps.state.memory` (mirroring the existing tools).

### 4. Refactor: extract `_fetch_entity_detail`

The HTTP-fetch body of the existing `fetch_entity_details` (`result_actions.py:54-99` — build URL, attach bearer token, 401/403 refresh-and-retry, 404 → `ModelRetry`, `raise_for_status`, build `DataArtifact`) is extracted into a private helper:

```python
async def _fetch_entity_detail(
    ctx: RunContext[StateDeps[SearchState]],
    entity_id: str,
    entity_type: EntityType,
) -> ToolReturn:
    ...
```

Both `fetch_entity_details` (unchanged public contract — still takes a known UUID) and `get_entity_by_id`'s single-match/full-UUID paths call it. One auth/refresh path, no duplication.

### 5. Planner routing & prompts

No code-level intent detection. The change is prompt guidance only:

- **`get_planning_prompt`** (`prompts.py:180`): extend the existing note so it explicitly covers id/prefix — e.g. "Getting detailed data for a single entity by its id or id-prefix (when the entity type is stated) requires a single RESULT_ACTIONS task, not SEARCH."
- **`get_result_actions_prompt`** (`prompts.py:218-220`): add an action line — "If the user references a specific entity by id or id-prefix: Call `{tools.get_entity_by_id.__name__}(id_or_prefix=..., entity_type=...)`" — and clarify that `fetch_entity_details` is for when a full known UUID is already in hand.
- **`Task.action_type` description** (`state.py:46`): no change required (RESULT_ACTIONS already covers "get detailed data for a single entity"), but verify wording still reads correctly.

`get_entity_by_id` must be exported from `tools/__init__.py` (added to imports + `__all__`) so `prompts.py` can reference `tools.get_entity_by_id.__name__`.

## Error handling & scope

- Invalid (non-hex) input, too-short prefix, and zero matches all raise `ModelRetry` with actionable text, letting the skill LLM recover (ask the user, or fall back to search).
- Full-UUID 404 keeps the existing `ModelRetry` from `_fetch_entity_detail`.
- DB errors propagate; `SkillRunner` already records a failing `ToolStep` and the planner stops the task (`planner.py:102-110`).
- The DB prefix query uses `cast(id_col, Text) ilike '<prefix>%'`; on large tables this is a sequential scan, but bounded by `LIMIT N+1` and acceptable for interactive use. (No new index introduced; flag if it becomes a hotspot.)
- Resolution reads `db.session` directly, consistent with the search tools (`tools/search.py:165`).

## Tests

- **`tests/test_entity_lookup.py`** — the pure pieces, no real DB:
  - `_ENTITY_LOOKUP` has an entry for every `EntityType` (guards against a missing mapping).
  - id-form classification helpers (`_classify_id`: full UUID vs. valid prefix vs. too-short vs. non-hex), `@pytest.mark.parametrize`'d with `pytest.param(..., id=...)`.
  - `resolve_entity_id_prefix` with a **fake session** (a `SimpleNamespace`/stub whose `query(...).filter(...).limit(...).all()` returns canned rows) — asserts the right table/columns are selected and rows map to `ResolvedEntity`. (Real prefix-matching SQL needs a DB and is out of scope for unit tests; note this in the test module docstring.)
- **`tests/test_result_actions.py`** — `get_entity_by_id` decision logic, with `resolve_entity_id_prefix` and `_fetch_entity_detail` monkeypatched and `db` faked via `SimpleNamespace(session=object())` (mirroring `tests/test_search.py:192`):
  - single match / full UUID → delegates to `_fetch_entity_detail` (assert it was called with the resolved UUID; `_fetch_entity_detail` stubbed to return a sentinel `ToolReturn` with a `DataArtifact`);
  - 2+ matches → plain `ToolReturn` whose `metadata` is `None` and whose `return_value` lists the candidates;
  - 0 matches / too-short / non-hex → raises `ModelRetry`.
- No planner-LLM test (matches repo convention; the planner change is prompt text + the full suite staying green).

## Out of scope

- Deterministic pre-planner shortcut that skips the planner LLM entirely (possible later optimization layered on top of this tool; the `execute(target_action=...)` path at `planner.py:138` already exists to support it).
- Type inference for a bare id with no stated entity type.
- Cross-entity-type prefix resolution (resolving a prefix against all tables at once).
- Changing `fetch_entity_details`'s public contract or the export flow.
- Adding a DB index for id-prefix lookups.
