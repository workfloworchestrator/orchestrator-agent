# Fetch-entity-by-id fast path (`get_entity_by_id`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `get_entity_by_id` Result Actions tool that resolves an entity from a full UUID *or* a partial id-prefix in a single skill run, fetching full details on an unambiguous match and returning a candidate list otherwise.

**Architecture:** A new pure module `entity_lookup.py` owns the EntityType→table mapping, id-form classification, and a direct DB prefix resolver. The `result_actions.py` tool dispatches on the classified id form: full UUID / single prefix match → HTTP detail fetch (extracted into a shared `_fetch_entity_detail` helper); 2+ matches → a plain `ToolReturn` candidate list; zero/too-short/non-hex → `ModelRetry` so the skill LLM can recover. Planner routing is prompt-only — no code-level intent detection.

**Tech Stack:** Python 3.11–3.13, pydantic-ai (`FunctionToolset`, `ToolReturn`, `ModelRetry`, `StateDeps`), orchestrator-core 5.0.2 (`EntityType`, `db.session`/`WrappedSession`, `SubscriptionTable`/`ProductTable`/`WorkflowTable`/`ProcessTable`), SQLAlchemy (`func`, `cast`, `Text`), pytest + pytest-asyncio (asyncio_mode=auto), ruff (line-length 120), mypy (strict).

---

## Background facts (verified against the live env, do not re-verify)

- **Table classes** live at `orchestrator.core.db.models`. Confirmed columns:
  | EntityType | Table | id column | title |
  |---|---|---|---|
  | `SUBSCRIPTION` | `SubscriptionTable` | `subscription_id` | `description` |
  | `PRODUCT` | `ProductTable` | `product_id` | `name` |
  | `WORKFLOW` | `WorkflowTable` | `workflow_id` | `name` |
  | `PROCESS` | `ProcessTable` | `pid` | computed `f"{workflow_id} ({last_status})"` (no name column) |
- `EntityType` has exactly 4 members: `SUBSCRIPTION`, `PRODUCT`, `WORKFLOW`, `PROCESS`. `EntityType.SUBSCRIPTION.value == "SUBSCRIPTION"` (uppercase).
- `db` is imported as `from orchestrator.core.db import db`; `db.session` is a `WrappedSession` (imported as `from orchestrator.core.db.database import WrappedSession`). `WrappedSession.query(...)` exists (legacy Query API).
- `cast(SubscriptionTable.subscription_id, Text).ilike("abcd%")` compiles; `func.concat(cast(ProcessTable.workflow_id, Text), " (", ProcessTable.last_status, ")")` compiles.
- httpx-mock pattern in tests: `AsyncMock(spec=httpx.AsyncClient)` + set `__aenter__`/`__aexit__`, then `patch("<module>.httpx.AsyncClient", return_value=mock_client)` (see `tests/test_auth.py:46-51`).
- fake-session pattern: `monkeypatch.setattr(<module>, "db", SimpleNamespace(session=object()))` (see `tests/test_search.py:192`).
- mypy config has `warn_unreachable = true` and `warn_no_return = true`: **do not** add a `case _:` default to a `match` that already covers every `IdForm` member — it would be flagged unreachable. Enum-exhaustive `match` satisfies "always returns".

---

## File Structure

- **Create** `src/orchestrator_agent/entity_lookup.py` — EntityType→table mapping (`_ENTITY_LOOKUP`), `ResolvedEntity`, `_LookupSpec`, `IdForm`, `MIN_PREFIX_LEN`, `_classify_id`, `resolve_entity_id_prefix`. Pure (no LLM, no HTTP); the resolver takes a `session` argument so it is unit-testable with a fake.
- **Modify** `src/orchestrator_agent/tools/result_actions.py` — extract `_fetch_entity_detail` from `fetch_entity_details`; add the `get_entity_by_id` tool + `_resolve_prefix` helper; add `db` and `entity_lookup` imports.
- **Modify** `src/orchestrator_agent/tools/__init__.py` — import + `__all__` export `get_entity_by_id`.
- **Modify** `src/orchestrator_agent/prompts.py` — extend the planner note and the result-actions action list.
- **Create** `tests/test_entity_lookup.py` — mapping completeness, `_classify_id` parametrized, `resolve_entity_id_prefix` with a fake session.
- **Create** `tests/test_result_actions.py` — `get_entity_by_id` decision logic (resolver + `_fetch_entity_detail` stubbed) and a `fetch_entity_details` refactor-safety test (httpx mocked).
- **Modify** `tests/test_prompts.py` — assert `get_entity_by_id` appears in the result-actions prompt and id-prefix wording in the planner prompt.

---

## Task 1: `entity_lookup.py` — mapping + id classification

**Files:**
- Create: `src/orchestrator_agent/entity_lookup.py`
- Test: `tests/test_entity_lookup.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_entity_lookup.py`:

```python
# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the entity_lookup module.

Real prefix-matching SQL requires a live database and is out of scope here; the
resolver test uses a fake session that records the query arguments and returns
canned rows.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from orchestrator.core.search.core.types import EntityType

from orchestrator_agent.entity_lookup import (
    _ENTITY_LOOKUP,
    IdForm,
    _classify_id,
)


def test_entity_lookup_covers_all_entity_types():
    for entity_type in EntityType:
        assert entity_type in _ENTITY_LOOKUP, f"missing lookup spec for {entity_type}"


class TestClassifyId:
    @pytest.mark.parametrize(
        "raw, expected_form, expected_norm",
        [
            pytest.param(
                "12345678-1234-5678-1234-567812345678",
                IdForm.FULL_UUID,
                "12345678-1234-5678-1234-567812345678",
                id="full-uuid",
            ),
            pytest.param(
                "12345678-1234-5678-1234-567812345678".upper(),
                IdForm.FULL_UUID,
                "12345678-1234-5678-1234-567812345678",
                id="full-uuid-uppercase-normalized",
            ),
            pytest.param("  abcd1234  ", IdForm.PREFIX, "abcd1234", id="prefix-stripped"),
            pytest.param("ABCD1234", IdForm.PREFIX, "abcd1234", id="prefix-lowercased"),
            pytest.param("abcd", IdForm.PREFIX, "abcd", id="prefix-min-length"),
            pytest.param("abc", IdForm.TOO_SHORT, "abc", id="too-short-3-hex"),
            pytest.param("ab-c", IdForm.TOO_SHORT, "ab-c", id="too-short-hyphen-ignored"),
            pytest.param("acme", IdForm.NON_HEX, "acme", id="non-hex-letter"),
            pytest.param("hello world", IdForm.NON_HEX, "hello world", id="non-hex-space"),
            pytest.param("", IdForm.NON_HEX, "", id="empty"),
        ],
    )
    def test_classify(self, raw, expected_form, expected_norm):
        form, norm = _classify_id(raw)
        assert form is expected_form
        assert norm == expected_norm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_entity_lookup.py -v`
Expected: collection/import error — `ModuleNotFoundError: No module named 'orchestrator_agent.entity_lookup'`.

- [ ] **Step 3: Create the module with mapping + classification**

Create `src/orchestrator_agent/entity_lookup.py`:

```python
# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct-DB resolution of an entity id or id-prefix to (id, title).

This module is intentionally free of LLM/HTTP concerns so the resolver and the
id-form classifier are unit-testable in isolation. ``resolve_entity_id_prefix``
runs a cheap, deterministic prefix query against orchestrator-core tables — it
is not the semantic search engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

from orchestrator.core.db.database import WrappedSession
from orchestrator.core.db.models import (
    ProcessTable,
    ProductTable,
    SubscriptionTable,
    WorkflowTable,
)
from orchestrator.core.search.core.types import EntityType
from sqlalchemy import Text, cast, func

MIN_PREFIX_LEN = 4
_HEX_CHARS = set("0123456789abcdef-")


@dataclass(frozen=True)
class ResolvedEntity:
    """A single id-prefix match: the full id and a human-readable title."""

    entity_id: str
    title: str


@dataclass(frozen=True)
class _LookupSpec:
    """Per-EntityType id column and title expression for a prefix query."""

    id_col: Any  # SQLAlchemy InstrumentedAttribute (the id/primary-key column)
    title_expr: Any  # SQLAlchemy column or computed expression yielding the title


_ENTITY_LOOKUP: dict[EntityType, _LookupSpec] = {
    EntityType.SUBSCRIPTION: _LookupSpec(SubscriptionTable.subscription_id, SubscriptionTable.description),
    EntityType.PRODUCT: _LookupSpec(ProductTable.product_id, ProductTable.name),
    EntityType.WORKFLOW: _LookupSpec(WorkflowTable.workflow_id, WorkflowTable.name),
    EntityType.PROCESS: _LookupSpec(
        ProcessTable.pid,
        func.concat(cast(ProcessTable.workflow_id, Text), " (", ProcessTable.last_status, ")"),
    ),
}


class IdForm(Enum):
    """Classification of a raw id-or-prefix input."""

    FULL_UUID = "full_uuid"
    PREFIX = "prefix"
    TOO_SHORT = "too_short"
    NON_HEX = "non_hex"


def _classify_id(raw: str) -> tuple[IdForm, str]:
    """Normalize (strip + lower-case) an id-or-prefix and classify its form.

    Returns the classification and the normalized string. When the form is
    ``PREFIX`` the normalized string is the value to feed to a LIKE query.
    Length is measured in hex digits (hyphens ignored) against ``MIN_PREFIX_LEN``.
    """
    norm = raw.strip().lower()
    if not norm or any(char not in _HEX_CHARS for char in norm):
        return IdForm.NON_HEX, norm
    try:
        UUID(norm)
    except ValueError:
        if len(norm.replace("-", "")) < MIN_PREFIX_LEN:
            return IdForm.TOO_SHORT, norm
        return IdForm.PREFIX, norm
    return IdForm.FULL_UUID, norm
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_entity_lookup.py -v`
Expected: PASS (`test_entity_lookup_covers_all_entity_types` + all `TestClassifyId` params).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/entity_lookup.py tests/test_entity_lookup.py
git commit -m "Add entity_lookup mapping and id-form classifier"
```

---

## Task 2: `resolve_entity_id_prefix` resolver

**Files:**
- Modify: `src/orchestrator_agent/entity_lookup.py`
- Test: `tests/test_entity_lookup.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_entity_lookup.py` (and extend the import to include `ResolvedEntity` and `resolve_entity_id_prefix`):

Update the import block at the top of the file to:

```python
from orchestrator_agent.entity_lookup import (
    _ENTITY_LOOKUP,
    IdForm,
    ResolvedEntity,
    _classify_id,
    resolve_entity_id_prefix,
)
```

Append this test class:

```python
class _FakeQuery:
    """Records filter/limit calls and returns canned rows from .all()."""

    def __init__(self, recorder: dict, rows: list):
        self._recorder = recorder
        self._rows = rows

    def filter(self, *args):
        self._recorder["filter_called"] = True
        return self

    def limit(self, n):
        self._recorder["limit_arg"] = n
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Stand-in for WrappedSession that records query() column args."""

    def __init__(self, rows: list):
        self.rows = rows
        self.recorder: dict = {}

    def query(self, *args):
        self.recorder["query_args"] = args
        return _FakeQuery(self.recorder, self.rows)


class TestResolveEntityIdPrefix:
    def test_selects_id_and_title_columns_and_maps_rows(self):
        rows = [("11111111-aaaa", "Acme Corp"), ("11111111-bbbb", "Beta LLC")]
        session = _FakeSession(rows)

        result = resolve_entity_id_prefix(session, EntityType.SUBSCRIPTION, "1111", limit=10)

        # Columns selected match the spec for this entity type (identity, not ==,
        # because SQLAlchemy columns overload __eq__ to build SQL expressions).
        spec = _ENTITY_LOOKUP[EntityType.SUBSCRIPTION]
        assert session.recorder["query_args"][0] is spec.id_col
        assert session.recorder["query_args"][1] is spec.title_expr
        # limit+1 so the caller can detect "more than limit" without a count query.
        assert session.recorder["limit_arg"] == 11
        assert session.recorder["filter_called"] is True
        assert result == [
            ResolvedEntity(entity_id="11111111-aaaa", title="Acme Corp"),
            ResolvedEntity(entity_id="11111111-bbbb", title="Beta LLC"),
        ]

    def test_empty_rows_returns_empty_list(self):
        session = _FakeSession([])
        result = resolve_entity_id_prefix(session, EntityType.PRODUCT, "dead", limit=10)
        assert result == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_entity_lookup.py::TestResolveEntityIdPrefix -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_entity_id_prefix'`.

- [ ] **Step 3: Add the resolver to `entity_lookup.py`**

Append to `src/orchestrator_agent/entity_lookup.py`:

```python
def resolve_entity_id_prefix(
    session: WrappedSession,
    entity_type: EntityType,
    prefix: str,
    limit: int,
) -> list[ResolvedEntity]:
    """Return entities whose id starts with ``prefix`` (case-insensitive), up to ``limit``.

    Fetches ``limit + 1`` rows so the caller can detect "more than limit" matches
    without a second count query. The prefix is matched against the id cast to
    text; on large tables this is a sequential scan, bounded by the limit.
    """
    spec = _ENTITY_LOOKUP[entity_type]
    rows = (
        session.query(spec.id_col, spec.title_expr)
        .filter(cast(spec.id_col, Text).ilike(f"{prefix}%"))
        .limit(limit + 1)
        .all()
    )
    return [ResolvedEntity(entity_id=str(row[0]), title=str(row[1])) for row in rows]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_entity_lookup.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/entity_lookup.py tests/test_entity_lookup.py
git commit -m "Add resolve_entity_id_prefix direct-DB resolver"
```

---

## Task 3: Extract `_fetch_entity_detail` from `fetch_entity_details`

This is a behavior-preserving refactor: the HTTP fetch + auth-refresh + artifact-building body moves into a private helper that both `fetch_entity_details` and (later) `get_entity_by_id` call. A new httpx-mocked test guards the extraction.

**Files:**
- Modify: `src/orchestrator_agent/tools/result_actions.py:38-99`
- Test: `tests/test_result_actions.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_result_actions.py`:

```python
# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for result_actions tools (get_entity_by_id decision logic + fetch refactor)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import httpx
import pytest
from orchestrator.core.search.core.types import EntityType

import orchestrator_agent.tools.result_actions as ra
from orchestrator_agent.artifacts import DataArtifact
from orchestrator_agent.state import SearchState


def _ctx_with_active_step():
    """A RunContext-like object with memory ready to record a ToolStep."""
    state = SearchState(user_input="details please")
    state.memory.start_turn("details please")
    state.memory.start_step("ResultActions")
    return SimpleNamespace(deps=SimpleNamespace(state=state))


async def test_fetch_entity_details_returns_data_artifact(monkeypatch):
    ctx = _ctx_with_active_step()

    response = httpx.Response(
        200,
        json={"id": "e1", "status": "active"},
        request=httpx.Request("GET", "http://orchestrator/api/subscriptions/domain-model/e1"),
    )
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(
        ra, "token_manager", SimpleNamespace(get_token=AsyncMock(return_value=None), auth_enabled=False)
    )
    with patch("orchestrator_agent.tools.result_actions.httpx.AsyncClient", return_value=mock_client):
        result = await ra.fetch_entity_details(ctx, "e1", EntityType.SUBSCRIPTION)

    assert isinstance(result.metadata, DataArtifact)
    assert result.metadata.entity_id == "e1"
    assert result.metadata.entity_type == "SUBSCRIPTION"
    assert json.loads(result.return_value)["status"] == "active"


async def test_fetch_entity_details_404_raises_model_retry(monkeypatch):
    from pydantic_ai.exceptions import ModelRetry

    ctx = _ctx_with_active_step()

    response = httpx.Response(404, request=httpx.Request("GET", "http://orchestrator/api/subscriptions/domain-model/x"))
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(
        ra, "token_manager", SimpleNamespace(get_token=AsyncMock(return_value=None), auth_enabled=False)
    )
    with patch("orchestrator_agent.tools.result_actions.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ModelRetry):
            await ra.fetch_entity_details(ctx, "x", EntityType.SUBSCRIPTION)
```

- [ ] **Step 2: Run the test to verify it passes against current code, then we refactor**

Run: `uv run pytest tests/test_result_actions.py -v`
Expected: PASS — these tests describe the *current* behavior of `fetch_entity_details`. (They are the safety net; the refactor in Step 3 must keep them green.)

> Note: this is a refactor, so the test passes before the change. That is intentional — it pins behavior so the extraction can't silently regress.

- [ ] **Step 3: Extract the helper**

In `src/orchestrator_agent/tools/result_actions.py`, replace the whole `fetch_entity_details` function (lines 38-99) with a private helper plus a thin public tool that delegates:

```python
async def _fetch_entity_detail(
    ctx: RunContext[StateDeps[SearchState]],
    entity_id: str,
    entity_type: EntityType,
) -> ToolReturn:
    """Fetch the full domain model for a known UUID via the orchestrator HTTP API.

    The HTTP API assembles the full domain model (product blocks, instance
    values, etc.) that a flat DB row does not contain. Handles bearer-token
    auth with a single 401/403 refresh-and-retry; 404 becomes a ModelRetry.
    """
    logger.debug(
        "Fetching detailed entity data",
        entity_type=entity_type.value,
        entity_id=entity_id,
    )

    url = agent_settings.orchestrator_api_paths.entity_url(entity_type, entity_id)

    headers: dict[str, str] = {}
    token = await token_manager.get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=30)

        if response.status_code in (401, 403) and token_manager.auth_enabled:
            new_token = await token_manager.refresh_token()
            headers["Authorization"] = f"Bearer {new_token}"
            response = await client.get(url, headers=headers, timeout=30)

    if response.status_code == 404:
        raise ModelRetry(f"No {entity_type.value} found with ID {entity_id}.")
    response.raise_for_status()

    detailed = response.json()

    description = f"Fetched details for {entity_type.value} {entity_id}"

    ctx.deps.state.memory.record_tool_step(
        ToolStep(
            step_type="fetch_entity_details",
            description=description,
            context={"entity_id": entity_id},
        )
    )

    detailed_json = json.dumps(detailed, indent=2, default=str)

    artifact = DataArtifact(
        description=description,
        entity_id=entity_id,
        entity_type=entity_type.value,
    )

    return ToolReturn(return_value=detailed_json, metadata=artifact)


@result_actions_toolset.tool
async def fetch_entity_details(
    ctx: RunContext[StateDeps[SearchState]],
    entity_id: str,
    entity_type: EntityType,
) -> ToolReturn:
    """Fetch detailed information for a single entity by its known UUID.

    Use this when a full, exact UUID is already in hand (e.g. from a previous
    search result). To resolve a partial id-prefix, use get_entity_by_id instead.

    Args:
        ctx: Runtime context for agent (injected).
        entity_id: The exact UUID of the entity to fetch details for.
        entity_type: Type of entity.

    Returns:
        ToolReturn with entity JSON and DataArtifact metadata.
    """
    return await _fetch_entity_detail(ctx, entity_id, entity_type)
```

- [ ] **Step 4: Run the test to verify it still passes**

Run: `uv run pytest tests/test_result_actions.py -v`
Expected: PASS (behavior preserved).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/tools/result_actions.py tests/test_result_actions.py
git commit -m "Extract _fetch_entity_detail helper from fetch_entity_details"
```

---

## Task 4: `get_entity_by_id` tool + `_resolve_prefix` helper

**Files:**
- Modify: `src/orchestrator_agent/tools/result_actions.py` (add imports, `_PREFIX_MATCH_LIMIT`, `_resolve_prefix`, `get_entity_by_id`)
- Test: `tests/test_result_actions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_result_actions.py`:

```python
from pydantic_ai.exceptions import ModelRetry  # noqa: E402 (grouped with the new tests)

from orchestrator_agent.entity_lookup import ResolvedEntity  # noqa: E402

FULL_UUID = "12345678-1234-5678-1234-567812345678"


def _patch_db(monkeypatch):
    """get_entity_by_id reads db.session; the resolver is stubbed so the session is unused."""
    monkeypatch.setattr(ra, "db", SimpleNamespace(session=object()))


def _stub_fetch(monkeypatch):
    """Replace _fetch_entity_detail with a recording sentinel that returns a DataArtifact."""
    calls: list = []

    async def fake_fetch(ctx, entity_id, entity_type):
        calls.append((entity_id, entity_type))
        return ra.ToolReturn(
            return_value="SENTINEL",
            metadata=DataArtifact(description="d", entity_id=entity_id, entity_type=entity_type.value),
        )

    monkeypatch.setattr(ra, "_fetch_entity_detail", fake_fetch)
    return calls


class TestGetEntityById:
    async def test_full_uuid_delegates_to_fetch_without_resolving(self, monkeypatch):
        _patch_db(monkeypatch)
        calls = _stub_fetch(monkeypatch)
        resolve_calls: list = []
        monkeypatch.setattr(
            ra, "resolve_entity_id_prefix", lambda *a, **k: resolve_calls.append(a) or []  # noqa: E731
        )
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, FULL_UUID, EntityType.SUBSCRIPTION)

        assert result.return_value == "SENTINEL"
        assert calls == [(FULL_UUID, EntityType.SUBSCRIPTION)]
        assert resolve_calls == []  # full UUID skips resolution

    async def test_single_prefix_match_delegates_to_fetch(self, monkeypatch):
        _patch_db(monkeypatch)
        calls = _stub_fetch(monkeypatch)
        monkeypatch.setattr(
            ra, "resolve_entity_id_prefix", lambda *a, **k: [ResolvedEntity(FULL_UUID, "Acme")]
        )
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, "abcd1234", EntityType.SUBSCRIPTION)

        assert result.return_value == "SENTINEL"
        assert calls == [(FULL_UUID, EntityType.SUBSCRIPTION)]

    async def test_multiple_matches_return_plain_candidate_list(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        matches = [ResolvedEntity("aaaa1111", "Acme"), ResolvedEntity("aaaa2222", "Beta")]
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", lambda *a, **k: matches)
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, "aaaa", EntityType.SUBSCRIPTION)

        assert result.metadata is None  # plain ToolReturn, no artifact
        assert result.return_value["candidates"] == [
            {"entity_id": "aaaa1111", "title": "Acme"},
            {"entity_id": "aaaa2222", "title": "Beta"},
        ]
        assert "refine" in result.return_value["instruction"].lower()

    async def test_more_than_limit_matches_caps_and_notes(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        # Resolver returns limit+1 (11) rows to signal "more than limit".
        matches = [ResolvedEntity(f"aaaa{i:04d}", f"Title {i}") for i in range(11)]
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", lambda *a, **k: matches)
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, "aaaa", EntityType.SUBSCRIPTION)

        assert len(result.return_value["candidates"]) == 10  # capped
        assert "first 10" in result.return_value["instruction"]

    async def test_zero_matches_raises_model_retry(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", lambda *a, **k: [])
        ctx = _ctx_with_active_step()

        with pytest.raises(ModelRetry, match="No SUBSCRIPTION found with id starting with abcd"):
            await ra.get_entity_by_id(ctx, "abcd1234", EntityType.SUBSCRIPTION)

    async def test_too_short_prefix_raises_without_resolving(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        resolve_calls: list = []
        monkeypatch.setattr(
            ra, "resolve_entity_id_prefix", lambda *a, **k: resolve_calls.append(a) or []  # noqa: E731
        )
        ctx = _ctx_with_active_step()

        with pytest.raises(ModelRetry, match="at least 4 characters"):
            await ra.get_entity_by_id(ctx, "abc", EntityType.SUBSCRIPTION)
        assert resolve_calls == []

    async def test_non_hex_raises_search_hint(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", lambda *a, **k: [])
        ctx = _ctx_with_active_step()

        with pytest.raises(ModelRetry, match="search by name"):
            await ra.get_entity_by_id(ctx, "acme corp", EntityType.SUBSCRIPTION)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_result_actions.py::TestGetEntityById -v`
Expected: FAIL — `AttributeError: module 'orchestrator_agent.tools.result_actions' has no attribute 'get_entity_by_id'` (and no `resolve_entity_id_prefix`/`db` attributes yet).

- [ ] **Step 3: Add imports, constant, helper, and tool**

In `src/orchestrator_agent/tools/result_actions.py`, add these imports next to the existing ones (after the `from orchestrator...EntityType` line and the `orchestrator_agent` imports):

```python
from orchestrator.core.db import db
```

and

```python
from orchestrator_agent.entity_lookup import (
    IdForm,
    _classify_id,
    resolve_entity_id_prefix,
)
```

Add a module-level constant near the top (after `result_actions_toolset = ...`):

```python
_PREFIX_MATCH_LIMIT = 10
```

Add the helper and the tool (place them after `_fetch_entity_detail`/`fetch_entity_details`):

```python
async def _resolve_prefix(
    ctx: RunContext[StateDeps[SearchState]],
    entity_type: EntityType,
    prefix: str,
) -> ToolReturn:
    """Resolve a partial id-prefix: fetch on a unique match, else list candidates."""
    matches = resolve_entity_id_prefix(db.session, entity_type, prefix, limit=_PREFIX_MATCH_LIMIT)

    if not matches:
        raise ModelRetry(f"No {entity_type.value} found with id starting with {prefix}.")

    if len(matches) == 1:
        return await _fetch_entity_detail(ctx, matches[0].entity_id, entity_type)

    capped = matches[:_PREFIX_MATCH_LIMIT]
    candidates = [{"entity_id": m.entity_id, "title": m.title} for m in capped]
    instruction = (
        f"Multiple {entity_type.value} ids start with {prefix}. "
        "Ask the user to refine the id or pick one of these."
    )
    if len(matches) > _PREFIX_MATCH_LIMIT:
        instruction += f" Showing the first {_PREFIX_MATCH_LIMIT} — refine the id for fewer."

    description = f"Found {len(capped)} {entity_type.value} candidates for id prefix {prefix}"
    ctx.deps.state.memory.record_tool_step(
        ToolStep(
            step_type="get_entity_by_id",
            description=description,
            context={"prefix": prefix, "match_count": len(capped)},
        )
    )

    return ToolReturn(return_value={"candidates": candidates, "instruction": instruction})


@result_actions_toolset.tool
async def get_entity_by_id(
    ctx: RunContext[StateDeps[SearchState]],
    id_or_prefix: str,
    entity_type: EntityType,
) -> ToolReturn:
    """Resolve a specific entity by full UUID or partial id-prefix, then fetch its details.

    Use this when the user references a specific entity by its id — including just
    the first few characters they copied — and the entity type is stated. On a
    unique match the full domain model is fetched; on multiple matches a candidate
    list is returned for the user to disambiguate.

    Args:
        ctx: Runtime context for agent (injected).
        id_or_prefix: A full UUID or a partial id-prefix (at least 4 hex characters).
        entity_type: Type of entity (SUBSCRIPTION, PRODUCT, WORKFLOW, PROCESS).

    Returns:
        ToolReturn with entity details (DataArtifact) for a unique match, or a plain
        candidate list when the prefix is ambiguous.
    """
    form, normalized = _classify_id(id_or_prefix)

    match form:
        case IdForm.NON_HEX:
            raise ModelRetry(f"'{id_or_prefix.strip()}' is not a UUID; search by name instead.")
        case IdForm.TOO_SHORT:
            raise ModelRetry("Need at least 4 characters of the id to look it up.")
        case IdForm.FULL_UUID:
            return await _fetch_entity_detail(ctx, normalized, entity_type)
        case IdForm.PREFIX:
            return await _resolve_prefix(ctx, entity_type, normalized)
```

> The `match` covers every `IdForm` member, so mypy treats it as exhaustive — do **not** add `case _:` (it would trip `warn_unreachable`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_result_actions.py -v`
Expected: PASS (refactor tests + all `TestGetEntityById` params).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/tools/result_actions.py tests/test_result_actions.py
git commit -m "Add get_entity_by_id tool with prefix resolution and disambiguation"
```

---

## Task 5: Export `get_entity_by_id` from the tools package

`prompts.py` references `tools.get_entity_by_id.__name__`, so the symbol must be importable as `orchestrator_agent.tools.get_entity_by_id`.

**Files:**
- Modify: `src/orchestrator_agent/tools/__init__.py:31-58`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_result_actions.py`:

```python
def test_get_entity_by_id_is_exported_from_tools_package():
    from orchestrator_agent import tools

    assert hasattr(tools, "get_entity_by_id")
    assert "get_entity_by_id" in tools.__all__
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_result_actions.py::test_get_entity_by_id_is_exported_from_tools_package -v`
Expected: FAIL — `AssertionError` (symbol not exported).

- [ ] **Step 3: Add the import and `__all__` entry**

In `src/orchestrator_agent/tools/__init__.py`, update the result_actions import block:

```python
from orchestrator_agent.tools.result_actions import (
    fetch_entity_details,
    get_entity_by_id,
    prepare_export,
    result_actions_toolset,
)
```

And add `"get_entity_by_id",` to `__all__` (keep it alphabetical-ish near `fetch_entity_details`):

```python
    "fetch_entity_details",
    "filter_building_toolset",
    "get_entity_by_id",
    "get_valid_operators",
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_result_actions.py::test_get_entity_by_id_is_exported_from_tools_package -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/tools/__init__.py tests/test_result_actions.py
git commit -m "Export get_entity_by_id from tools package"
```

---

## Task 6: Planner & result-actions prompt guidance

**Files:**
- Modify: `src/orchestrator_agent/prompts.py:180` (planner note) and `:218-220` (result-actions actions)
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_prompts.py`, extend `TestGetResultActionsPrompt` and `TestGetPlanningPrompt`:

```python
class TestGetResultActionsPrompt:
    def test_contains_key_elements(self):
        prompt = get_result_actions_prompt(_make_state())
        assert "Acting on Results" in prompt
        assert "prepare_export" in prompt
        assert "fetch_entity_details" in prompt

    def test_mentions_get_entity_by_id_for_id_or_prefix(self):
        prompt = get_result_actions_prompt(_make_state())
        assert "get_entity_by_id" in prompt
        assert "id-prefix" in prompt
```

And add to `TestGetPlanningPrompt`:

```python
    def test_mentions_id_prefix_single_result_actions_task(self):
        prompt = get_planning_prompt(_make_state())
        assert "id-prefix" in prompt
        assert "RESULT_ACTIONS" in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL on `test_mentions_get_entity_by_id_for_id_or_prefix` and `test_mentions_id_prefix_single_result_actions_task` (strings not present yet).

- [ ] **Step 3: Update the prompts**

In `src/orchestrator_agent/prompts.py`, replace the planner note line (currently line 180) inside `guidelines`:

```python
        Note: Getting detailed data for a single entity by its id or id-prefix (when the entity type is stated) requires a single RESULT_ACTIONS task, not SEARCH. Preparing an export also requires a RESULT_ACTIONS task."""
```

In `get_result_actions_prompt`, replace the `## Available Actions` block (currently lines 218-220):

```python
        ## Available Actions
        - If user wants to EXPORT/DOWNLOAD results: Call {tools.prepare_export.__name__}() ONLY
        - If the user references a specific entity by id or id-prefix: Call {tools.get_entity_by_id.__name__}(id_or_prefix=..., entity_type=...)
        - If a full known UUID is already in hand (e.g. from a previous result): Call {tools.fetch_entity_details.__name__}(entity_id=..., entity_type=...)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS (existing + new prompt assertions).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator_agent/prompts.py tests/test_prompts.py
git commit -m "Route id/id-prefix detail lookups to get_entity_by_id in prompts"
```

---

## Task 7: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (no regressions in adapters/search/prompts).

- [ ] **Step 2: Lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors. (If `ruff format --check` reports diffs, run `uv run ruff format .` and re-stage.)

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/`
Expected: `Success: no issues found`. Watch specifically for: missing-return on `get_entity_by_id` (should be fine — enum-exhaustive `match`), and any `Any`-related complaints from `_LookupSpec` (typed `Any` deliberately).

- [ ] **Step 4: Commit any formatting fixes (only if Step 2 changed files)**

```bash
git add -u
git commit -m "Apply ruff formatting"
```

---

## Self-Review (completed during plan authoring)

**Spec coverage**

| Spec section | Task |
|---|---|
| §2 `entity_lookup.py` module, `_ENTITY_LOOKUP`, `ResolvedEntity`, `MIN_PREFIX_LEN`, `resolve_entity_id_prefix` | Tasks 1–2 |
| §3 `get_entity_by_id` tool, normalize/validate, full-UUID / prefix / 0 / 1 / 2…N / >N branches, ModelRetry texts, plain candidate ToolReturn, ToolStep recording | Task 4 |
| §3 "candidate list carries no artifact" (metadata None) | Task 4 (`test_multiple_matches_return_plain_candidate_list` asserts `metadata is None`) |
| §4 extract `_fetch_entity_detail`, both callers share it | Task 3 (+ single-match path in Task 4) |
| §5 planner prompt, result-actions prompt, export from `tools/__init__.py` | Tasks 5–6 |
| §5 `Task.action_type` description (verify only, no change) | No change needed — RESULT_ACTIONS description at `state.py:46` already reads "get detailed data for a single entity"; left as-is |
| Tests: `test_entity_lookup.py` (mapping completeness, `_classify_id`, fake-session resolver) | Tasks 1–2 |
| Tests: `test_result_actions.py` (single/full → fetch; 2+ → plain list; 0/short/non-hex → ModelRetry) | Task 4 |
| Error handling & scope (ModelRetry recovery, 404, DB errors propagate, `db.session` direct read) | Tasks 3–4 |

**Placeholder scan:** none — every code step contains complete code; every run step has an exact command + expected output.

**Type consistency:** `ResolvedEntity(entity_id, title)`, `IdForm.{FULL_UUID,PREFIX,TOO_SHORT,NON_HEX}`, `_classify_id(raw) -> (IdForm, str)`, `resolve_entity_id_prefix(session, entity_type, prefix, limit) -> list[ResolvedEntity]`, `_fetch_entity_detail(ctx, entity_id, entity_type)`, `_resolve_prefix(ctx, entity_type, prefix)`, `_PREFIX_MATCH_LIMIT = 10`, `MIN_PREFIX_LEN = 4` — names and signatures are consistent across Tasks 1, 2, 4 and the tests.

**Out of scope (per spec):** no pre-planner deterministic shortcut, no type inference for bare ids, no cross-entity prefix resolution, no change to `fetch_entity_details`'s public contract, no DB index.
