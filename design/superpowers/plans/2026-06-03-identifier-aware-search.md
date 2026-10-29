# Identifier-aware search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make search identifier-aware: a configurable env-driven domain-knowledge prompt section, generic identifier-extraction guidance, and an LLM-selectable, embedding-aware retriever (HYBRID for identifiers, SEMANTIC for sentences, FUZZY when embeddings are off).

**Architecture:** Three coordinated, mostly-independent changes. (1) A new `AGENT_DOMAIN_CONTEXT` setting injected as a `## Domain Knowledge` section into the search prompt. (2) An `EXTRACT IDENTIFIERS` rule in the shared `FILTERING_RULES`. (3) A `retriever` parameter on `run_search` threaded into the `SelectQuery`, gated by a pure `_effective_retriever` helper that degrades SEMANTIC/HYBRID → FUZZY when `llm_settings.EMBEDDING_API_ENABLED` is false, with prompt guidance conditioned on the same flag. Default `retriever=None` preserves today's auto-routing (backward compatible).

**Tech Stack:** Python 3.11–3.13, pydantic-ai (`FunctionToolset`, `ToolReturn`, `RunContext`, `StateDeps`), orchestrator-core 5.0.2 (`RetrieverType`, `SelectQuery`, `llm_settings.EMBEDDING_API_ENABLED`, search engine auto-routing), pydantic-settings (`AgentSettings`), pytest + pytest-asyncio (asyncio_mode=auto), ruff (line-length 120), mypy (strict).

---

## Background facts (verified against the live env — do not re-verify)

- `RetrieverType` (`orchestrator.core.search.core.types`) is `class RetrieverType(str, Enum)` with members `FUZZY="fuzzy"`, `SEMANTIC="semantic"`, `HYBRID="hybrid"`. Already imported in `src/orchestrator_agent/tools/search.py:20`.
- Embedding flag: `from orchestrator.core.settings import llm_settings` → `llm_settings.EMBEDDING_API_ENABLED` (a `bool`; `False` in the dev/test env).
- Auto-routing (`retriever is None`) picks SEMANTIC for multi-word `query_text`, FUZZY for a single token, HYBRID for a single non-UUID word (`retrieval/retrievers/base.py:_plan`). The agent always sets `query_text = state.user_input` (`tools/filters.py:75`), so identifier-bearing sentences get SEMANTIC today.
- `SelectQuery` has a validator: `retriever` may only be non-`None` when `query_text` is set (`query/mixins.py:validate_retriever_requires_query_text`). Setting `retriever=None` is always valid.
- An explicit SEMANTIC/HYBRID retriever with embeddings disabled raises `ValueError` in `Retriever.route` (`base.py:148-151`). The existing semantic *fallback* already catches this and degrades to auto/fuzzy — leave the fallback untouched.
- `get_search_execution_prompt` and `get_aggregation_execution_prompt` both embed the shared module-level `FILTERING_RULES` f-string (`src/orchestrator_agent/prompts.py:24-35`).
- Repo conventions: tests `uv run pytest <path> -v`; lint `uv run ruff check <paths>`; format `uv run ruff format <paths>` (black NOT installed). mypy strict via pre-commit (slow — let it finish). Commit with prefix `PATH="$HOME/.local/bin:$PATH"`; NO `Co-Authored-By` lines; on hook failure fix the root cause and make a NEW commit (never `--no-verify`, never amend). Source/test files start with the SURF/GÉANT Apache-2.0 header (copy from any existing file).
- Project test style: data-only variations MUST use `@pytest.mark.parametrize` with `pytest.param(..., id=...)`. Distinct behaviors may be separate test functions.

## File Structure

- **Modify** `src/orchestrator_agent/settings.py` — add `AGENT_DOMAIN_CONTEXT` field to `AgentSettings`.
- **Modify** `src/orchestrator_agent/prompts.py` — add `EXTRACT IDENTIFIERS` bullet to `FILTERING_RULES`; add `_domain_context_section()` + `_retriever_guidance()` helpers and inject them into `get_search_execution_prompt`; add `agent_settings` + `llm_settings` imports.
- **Modify** `src/orchestrator_agent/tools/search.py` — add `_effective_retriever()` helper + `llm_settings` import; add `retriever` param to `run_search` and `_execute_search_with_fallback`.
- **Modify** `tests/test_settings.py`, `tests/test_prompts.py`, `tests/test_search.py` — tests for each piece.

---

## Task 1: `AGENT_DOMAIN_CONTEXT` setting

**Files:**
- Modify: `src/orchestrator_agent/settings.py` (in `AgentSettings`, after the `AGENT_DEBUG` field at line 67)
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_settings.py`:

```python
def test_agent_domain_context_defaults_to_empty():
    assert AgentSettings().AGENT_DOMAIN_CONTEXT == ""


def test_agent_domain_context_accepts_value():
    assert AgentSettings(AGENT_DOMAIN_CONTEXT="circuit codes map to imsCircuitId").AGENT_DOMAIN_CONTEXT == (
        "circuit codes map to imsCircuitId"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings.py -k domain_context -v`
Expected: FAIL — `AgentSettings(AGENT_DOMAIN_CONTEXT=...)` raises (pydantic `init_forbid_extra`) / attribute missing.

- [ ] **Step 3: Add the field**

In `src/orchestrator_agent/settings.py`, add this field to `AgentSettings` immediately after the `AGENT_DEBUG` field (`AGENT_DEBUG: bool = Field(default=False, ...)`):

```python
    AGENT_DOMAIN_CONTEXT: str = Field(
        default="",
        description="Optional operator-supplied domain knowledge injected into the search prompt "
        "(e.g. identifier conventions and their filter fields). Empty disables the section.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings.py -k domain_context -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/settings.py tests/test_settings.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Add AGENT_DOMAIN_CONTEXT setting"
```

---

## Task 2: `EXTRACT IDENTIFIERS` guidance in `FILTERING_RULES`

**Files:**
- Modify: `src/orchestrator_agent/prompts.py:24-35` (the `FILTERING_RULES` f-string)
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_prompts.py`, the class `TestGetSearchExecutionPrompt.test_prompt_contains` is already parametrized with a `required` list. Add one parametrize case to it:

```python
            pytest.param(["EXTRACT IDENTIFIERS", "highest-signal"], id="identifier-extraction"),
```

(Insert it inside the existing `@pytest.mark.parametrize("required", [...])` list for `test_prompt_contains`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest "tests/test_prompts.py::TestGetSearchExecutionPrompt::test_prompt_contains[identifier-extraction]" -v`
Expected: FAIL — snippet not present yet.

- [ ] **Step 3: Add the bullet to `FILTERING_RULES`**

In `src/orchestrator_agent/prompts.py`, inside the `FILTERING_RULES` f-string, insert a new bullet directly BEFORE the `- Temporal constraints ...` line. Replace:

```python
- Temporal constraints like "in 2025", "between X and Y" require filters on datetime fields
```

with:

```python
- **EXTRACT IDENTIFIERS**: Scan the request for specific identifiers the user gave — entity/subscription ids, customer names, reference codes (e.g. `IS4443`), or numbers (e.g. `4433`, `id 1234`). These are the highest-signal part of the request. Use `{tools.discover_filter_paths.__name__}` to find the field that holds such a value and filter it with `like` (substring/typo-tolerant). If no discovered field clearly matches an opaque identifier, do NOT invent a filter — the search already ranks on the full request text. Never silently ignore an identifier the user provided.
- Temporal constraints like "in 2025", "between X and Y" require filters on datetime fields
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest "tests/test_prompts.py::TestGetSearchExecutionPrompt::test_prompt_contains[identifier-extraction]" -v`
Expected: PASS. Also run `uv run pytest tests/test_prompts.py -q` — all existing prompt tests still pass (the bullet also appears in the aggregation prompt, which has no conflicting assertion).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/prompts.py tests/test_prompts.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Add identifier-extraction guidance to filtering rules"
```

---

## Task 3: `_effective_retriever` embedding-aware helper

**Files:**
- Modify: `src/orchestrator_agent/tools/search.py` (add `llm_settings` import; add helper after `_describe_results`, ~line 49)
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_search.py`:

```python
class TestEffectiveRetriever:
    @pytest.mark.parametrize(
        "embeddings_enabled, requested, expected",
        [
            pytest.param(True, RetrieverType.HYBRID, RetrieverType.HYBRID, id="on-hybrid"),
            pytest.param(True, RetrieverType.SEMANTIC, RetrieverType.SEMANTIC, id="on-semantic"),
            pytest.param(True, RetrieverType.FUZZY, RetrieverType.FUZZY, id="on-fuzzy"),
            pytest.param(True, None, None, id="on-auto"),
            pytest.param(False, RetrieverType.HYBRID, RetrieverType.FUZZY, id="off-hybrid-degrades"),
            pytest.param(False, RetrieverType.SEMANTIC, RetrieverType.FUZZY, id="off-semantic-degrades"),
            pytest.param(False, RetrieverType.FUZZY, RetrieverType.FUZZY, id="off-fuzzy"),
            pytest.param(False, None, None, id="off-auto"),
        ],
    )
    def test_effective(self, monkeypatch, embeddings_enabled, requested, expected):
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", embeddings_enabled)
        assert search_mod._effective_retriever(requested) == expected
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_search.py::TestEffectiveRetriever -v`
Expected: FAIL — `module 'orchestrator_agent.tools.search' has no attribute 'llm_settings'` / `_effective_retriever`.

- [ ] **Step 3: Add the import and helper**

In `src/orchestrator_agent/tools/search.py`, add the import next to the other `orchestrator.core...` imports (isort will order it; place after `from orchestrator.core.db.database import WrappedSession`):

```python
from orchestrator.core.settings import llm_settings
```

Add the helper immediately after `_describe_results` (after its `return` at line 49):

```python
def _effective_retriever(requested: RetrieverType | None) -> RetrieverType | None:
    """Resolve the retriever to actually use, accounting for embedding availability.

    SEMANTIC and HYBRID need embeddings; when EMBEDDING_API_ENABLED is False they would
    raise ValueError in Retriever.route. Degrade those to FUZZY (which still keyword-matches
    the identifier). FUZZY and None (auto-routing) pass through unchanged.
    """
    if requested in (RetrieverType.SEMANTIC, RetrieverType.HYBRID) and not llm_settings.EMBEDDING_API_ENABLED:
        return RetrieverType.FUZZY
    return requested
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_search.py::TestEffectiveRetriever -v`
Expected: PASS (8 cases).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/tools/search.py tests/test_search.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Add embedding-aware _effective_retriever helper"
```

---

## Task 4: Thread `retriever` through `run_search`

**Files:**
- Modify: `src/orchestrator_agent/tools/search.py` (`_execute_search_with_fallback` at lines 113-125; `run_search` at lines 149-165)
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_search.py` inside `class TestExecuteSearchWithFallback` (a new parametrized method):

```python
    @pytest.mark.parametrize(
        "embeddings_enabled, requested, expected",
        [
            pytest.param(True, RetrieverType.HYBRID, RetrieverType.HYBRID, id="hybrid-on"),
            pytest.param(False, RetrieverType.HYBRID, RetrieverType.FUZZY, id="hybrid-off-degrades"),
            pytest.param(True, None, None, id="auto-passthrough"),
        ],
    )
    async def test_primary_pass_uses_effective_retriever(self, monkeypatch, embeddings_enabled, requested, expected):
        captured = []

        async def fake(query, session, run_id):
            captured.append(query.retriever)
            return _response([_result()], "structured"), RUN_ID, STRUCT_QID

        monkeypatch.setattr(search_mod, "execute_search_with_persistence", fake)
        monkeypatch.setattr(search_mod.llm_settings, "EMBEDDING_API_ENABLED", embeddings_enabled)
        state = _state()
        await search_mod._execute_search_with_fallback(state, EntityType.SUBSCRIPTION, 10, object(), requested)
        assert captured[0] == expected
```

And add a module-level test (outside the class) verifying `run_search` forwards the argument:

```python
async def test_run_search_forwards_retriever(monkeypatch):
    from types import SimpleNamespace

    from orchestrator.core.search.query.queries import SelectQuery

    captured = {}

    async def fake_execute(state, entity_type, limit, session, retriever):
        captured["retriever"] = retriever
        state.query_id = STRUCT_QID
        final_query = SelectQuery(entity_type=EntityType.SUBSCRIPTION, query_text="acme", filters=None, limit=10)
        return _response([_result("e1")], "structured"), final_query, False

    monkeypatch.setattr(search_mod, "_execute_search_with_fallback", fake_execute)
    monkeypatch.setattr(search_mod, "db", SimpleNamespace(session=object()))

    state = SearchState(user_input="acme", run_id=RUN_ID)
    state.query_id = STRUCT_QID
    state.memory.start_turn("acme")
    state.memory.start_step("Search")
    ctx = SimpleNamespace(deps=SimpleNamespace(state=state))

    await search_mod.run_search(ctx, EntityType.SUBSCRIPTION, 10, RetrieverType.HYBRID)
    assert captured["retriever"] == RetrieverType.HYBRID
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -k "primary_pass_uses_effective_retriever or run_search_forwards_retriever" -v`
Expected: FAIL — `_execute_search_with_fallback()` / `run_search()` take no `retriever` argument (TypeError).

- [ ] **Step 3: Add the `retriever` parameter and thread it**

In `src/orchestrator_agent/tools/search.py`, change `_execute_search_with_fallback`'s signature and its first two body lines. Replace:

```python
async def _execute_search_with_fallback(
    state: SearchState,
    entity_type: EntityType,
    limit: int,
    session: WrappedSession,
) -> tuple[SearchResponse, SelectQuery, bool]:
    """Run the structured pass, then a semantic fallback when it returns zero rows.

    Updates state.run_id/query_id to the query that produced the returned rows.
    Returns (response, final_query, fallback_used).
    """
    ensure_query_initialized(state, entity_type)
    query = cast(SelectQuery, cast(Query, state.query).model_copy(update={"limit": limit}))
```

with:

```python
async def _execute_search_with_fallback(
    state: SearchState,
    entity_type: EntityType,
    limit: int,
    session: WrappedSession,
    retriever: RetrieverType | None = None,
) -> tuple[SearchResponse, SelectQuery, bool]:
    """Run the structured pass, then a semantic fallback when it returns zero rows.

    Updates state.run_id/query_id to the query that produced the returned rows.
    Returns (response, final_query, fallback_used).
    """
    ensure_query_initialized(state, entity_type)
    effective = _effective_retriever(retriever) if state.user_input else None
    query = cast(SelectQuery, cast(Query, state.query).model_copy(update={"limit": limit, "retriever": effective}))
```

Then change `run_search`'s signature, docstring, and the call. Replace:

```python
@search_execution_toolset.tool
async def run_search(
    ctx: RunContext[StateDeps[SearchState]],
    entity_type: EntityType,
    limit: int = 10,
) -> ToolReturn:
    """Execute a search to find and rank entities.

    Use this tool for SELECT action to find entities matching your criteria.
    For counting or computing statistics, use run_aggregation instead.

    If the structured search finds nothing, this automatically retries with a
    broader semantic search (filters dropped) so the user still gets the closest
    matches; the result description states when matches are approximate.
    """
    state = ctx.deps.state
    response, final_query, fallback_used = await _execute_search_with_fallback(state, entity_type, limit, db.session)
```

with:

```python
@search_execution_toolset.tool
async def run_search(
    ctx: RunContext[StateDeps[SearchState]],
    entity_type: EntityType,
    limit: int = 10,
    retriever: RetrieverType | None = None,
) -> ToolReturn:
    """Execute a search to find and rank entities.

    Use this tool for SELECT action to find entities matching your criteria.
    For counting or computing statistics, use run_aggregation instead.

    If the structured search finds nothing, this automatically retries with a
    broader semantic search (filters dropped) so the user still gets the closest
    matches; the result description states when matches are approximate.

    Args:
        retriever: Ranking strategy. HYBRID (semantic + fuzzy keyword) for queries centered on
            an identifier/code/name; SEMANTIC for descriptive phrases; FUZZY for exact tokens or
            when embeddings are unavailable. Omit to auto-route.
    """
    state = ctx.deps.state
    response, final_query, fallback_used = await _execute_search_with_fallback(
        state, entity_type, limit, db.session, retriever
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS — the new tests plus all existing `TestExecuteSearchWithFallback` / `TestRunSearchArtifact` tests (default `retriever=None` keeps the primary pass retriever `None`, so `seen_retrievers == [None, RetrieverType.SEMANTIC, None]` still holds).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/tools/search.py tests/test_search.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Let run_search choose an embedding-aware retriever"
```

---

## Task 5: Domain section + retriever guidance in the search prompt

**Files:**
- Modify: `src/orchestrator_agent/prompts.py` (add imports; add `_domain_context_section()` + `_retriever_guidance()` helpers; update `get_search_execution_prompt`)
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts.py`:

```python
class TestDomainContextSection:
    @pytest.mark.parametrize(
        "context, expect_section",
        [
            pytest.param("IS#### maps to imsCircuitId", True, id="set"),
            pytest.param("", False, id="empty"),
            pytest.param("   ", False, id="whitespace-only"),
        ],
    )
    def test_domain_section(self, monkeypatch, context, expect_section):
        monkeypatch.setattr("orchestrator_agent.prompts.agent_settings.AGENT_DOMAIN_CONTEXT", context)
        prompt = get_search_execution_prompt(_make_state())
        assert ("## Domain Knowledge" in prompt) is expect_section
        if expect_section:
            assert context.strip() in prompt


class TestRetrieverGuidance:
    @pytest.mark.parametrize(
        "embeddings_enabled, expect_hybrid",
        [
            pytest.param(True, True, id="embeddings-on"),
            pytest.param(False, False, id="embeddings-off"),
        ],
    )
    def test_retriever_guidance(self, monkeypatch, embeddings_enabled, expect_hybrid):
        monkeypatch.setattr("orchestrator_agent.prompts.llm_settings.EMBEDDING_API_ENABLED", embeddings_enabled)
        prompt = get_search_execution_prompt(_make_state())
        assert "CHOOSE A RETRIEVER" in prompt
        if expect_hybrid:
            assert "retriever=HYBRID" in prompt
        else:
            assert "retriever=FUZZY" in prompt
            assert "retriever=HYBRID" not in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -k "TestDomainContextSection or TestRetrieverGuidance" -v`
Expected: FAIL — `## Domain Knowledge` / `CHOOSE A RETRIEVER` not present; also `agent_settings`/`llm_settings` not yet attributes of `orchestrator_agent.prompts`.

- [ ] **Step 3: Add imports and helpers, then wire them into the prompt**

In `src/orchestrator_agent/prompts.py`, add these imports (ruff/isort will order them — `orchestrator.*` in the third-party group, `orchestrator_agent.*` in the first-party group):

```python
from orchestrator.core.settings import llm_settings
from orchestrator_agent.settings import agent_settings
```

Add these two helpers immediately above `def get_search_execution_prompt`:

```python
def _domain_context_section() -> str:
    """Return a '## Domain Knowledge' block when AGENT_DOMAIN_CONTEXT is set, else ''."""
    context = agent_settings.AGENT_DOMAIN_CONTEXT.strip()
    if not context:
        return ""
    return f"## Domain Knowledge\n{context}"


def _retriever_guidance() -> str:
    """Retriever-selection guidance for the search prompt, conditioned on embedding availability."""
    if llm_settings.EMBEDDING_API_ENABLED:
        return (
            "- **CHOOSE A RETRIEVER**: For identifier/code/name-centric requests pass "
            f"`retriever=HYBRID` to `{tools.run_search.__name__}`; for descriptive/sentence "
            "requests use `retriever=SEMANTIC`; omit it to auto-route."
        )
    return (
        "- **CHOOSE A RETRIEVER**: Embeddings are unavailable — use `retriever=FUZZY` for "
        f"identifier matching with `{tools.run_search.__name__}` and rely on filters. "
        "Do not request embedding-based retrievers."
    )
```

Replace the entire body of `get_search_execution_prompt` (from `context = state.memory.format_context_for_llm(state)` through the final `.strip()`) with:

```python
    context = state.memory.format_context_for_llm(state)
    domain_section = _domain_context_section()
    retriever_guidance = _retriever_guidance()

    return dedent(
        f"""
        # Searching

        {AGENT_CONTEXT}

        ## Your Task
        Execute a database search to answer the user's request.
        **IMPORTANT**: This query starts empty - previous query filters shown in history are NOT applied unless you rebuild them.

        ## Steps
        1. Determine the entity_type for this search (SUBSCRIPTION, PRODUCT, WORKFLOW, or PROCESS)
        2. If filters needed (almost always):
           a. Call `{tools.discover_filter_paths.__name__}(field_names=[...], entity_type=...)` to get valid paths
           b. Call `{tools.get_valid_operators.__name__}` to confirm valid operators for the field type
           c. Build FilterTree using ONLY the exact paths from step 2a
           d. Call `{tools.set_filter_tree.__name__}` with the validated FilterTree
        3. Call {tools.run_search.__name__}(entity_type=...) — you MUST pass entity_type
        4. Explain what you did in 1-2 sentences at most. DO NOT list the actual results, they are already shown to the user.

        {retriever_guidance}

        {domain_section}
        {FILTERING_RULES}

        **Note:** If a search returns no results, the system automatically retries with a broader semantic search (filters dropped) and shows the closest matches. Don't over-constrain your filters. If the result description says matches are approximate, briefly tell the user the results are the closest available rather than exact matches.

        ---

        {context}
    """
    ).strip()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS — the new `TestDomainContextSection` / `TestRetrieverGuidance` plus all existing prompt tests (the existing search-prompt snippet assertions are unchanged and still present).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/prompts.py tests/test_prompts.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Inject domain context and retriever guidance into search prompt"
```

---

## Task 6: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (no regressions in adapters/search/prompts/settings).

- [ ] **Step 2: Lint + format check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: `All checks passed!` and all files already formatted. (If format reports diffs, run `uv run ruff format .`, re-stage, and commit "Apply ruff formatting".)

- [ ] **Step 3: Manual confirmation note (no code)**

mypy already runs green per-commit via pre-commit; a separate full `mypy src/` is optional and slow — skip unless requested. Record that the three behaviors are wired: `AGENT_DOMAIN_CONTEXT` → `## Domain Knowledge`; `EXTRACT IDENTIFIERS` in filtering rules; `run_search(retriever=...)` threaded with embedding-aware degradation and embedding-conditioned prompt guidance.

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| Component 1 — `AGENT_DOMAIN_CONTEXT` setting | Task 1 |
| Component 1 — `## Domain Knowledge` injection (`_domain_context_section`) | Task 5 |
| Component 2 — `EXTRACT IDENTIFIERS` in `FILTERING_RULES` | Task 2 |
| Component 3 — `_effective_retriever` (embedding-aware degrade) | Task 3 |
| Component 3 — `retriever` param threaded through `run_search`/`_execute_search_with_fallback` | Task 4 |
| Component 3 — embedding-conditioned `_retriever_guidance` in search prompt | Task 5 |
| Error handling: empty `user_input` → retriever `None` | Task 4 (`_effective_retriever(...) if state.user_input else None`) |
| Error handling: embeddings off + SEMANTIC/HYBRID requested → FUZZY | Tasks 3 & 4 (tested in both) |
| Backward compatibility: default `retriever=None` = auto-routing | Task 4 (existing fallback tests stay green) |
| Tests: settings default, prompt sections, `_effective_retriever`, threading | Tasks 1–5 |
| Out of scope (aggregation injection, heuristic detection, query_text control, run_aggregation retriever, fallback changes, get_entity_by_id) | not implemented — correct |

**Placeholder scan:** None — every code step has complete code and exact commands; `TestRetrieverGuidance` uses a clean `if/else` assertion.

**Type consistency:** `_effective_retriever(requested: RetrieverType | None) -> RetrieverType | None`, `run_search(..., retriever: RetrieverType | None = None)`, `_execute_search_with_fallback(..., retriever: RetrieverType | None = None)`, `_domain_context_section() -> str`, `_retriever_guidance() -> str`, `agent_settings.AGENT_DOMAIN_CONTEXT: str`, `llm_settings.EMBEDDING_API_ENABLED: bool` — names/signatures consistent across Tasks 1–5 and the tests (`search_mod.llm_settings`, `orchestrator_agent.prompts.agent_settings`/`llm_settings`).
