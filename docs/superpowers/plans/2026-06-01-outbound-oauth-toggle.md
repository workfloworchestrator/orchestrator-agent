# Outbound OAuth Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let outgoing (client-credentials) OAuth be enabled independently of incoming request authentication, so `OAUTH2_ACTIVE=False` (incoming off) can run alongside outbound auth on.

**Architecture:** Add a tri-state `OAUTH2_OUTBOUND_ACTIVE` (`bool | None`) field to `AgentSettings`. `None` (default) means "follow `OAUTH2_ACTIVE`", preserving current behavior; `True`/`False` override it. `OAuthTokenManager.auth_enabled` resolves this at read-time. The incoming `AuthMiddleware` is untouched and continues to read only `oauth2lib_settings.OAUTH2_ACTIVE`.

**Tech Stack:** Python 3.11–3.13, pydantic-settings, pytest + pytest-asyncio, `oauth2_lib`.

Reference spec: `docs/superpowers/specs/2026-06-01-outbound-oauth-toggle-design.md`

---

### Task 1: Add `OAUTH2_OUTBOUND_ACTIVE` setting

**Files:**
- Modify: `src/orchestrator_agent/settings.py` (add field to `AgentSettings`, after `AGENT_DEBUG` at line 67)
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_settings.py`:

```python
def test_oauth2_outbound_active_defaults_to_none():
    settings = AgentSettings()
    assert settings.OAUTH2_OUTBOUND_ACTIVE is None


def test_oauth2_outbound_active_accepts_explicit_bool():
    assert AgentSettings(OAUTH2_OUTBOUND_ACTIVE=True).OAUTH2_OUTBOUND_ACTIVE is True
    assert AgentSettings(OAUTH2_OUTBOUND_ACTIVE=False).OAUTH2_OUTBOUND_ACTIVE is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PATH="$HOME/.local/bin:$PATH" uv run pytest tests/test_settings.py::test_oauth2_outbound_active_defaults_to_none tests/test_settings.py::test_oauth2_outbound_active_accepts_explicit_bool -v`
Expected: FAIL — `AgentSettings` has no field `OAUTH2_OUTBOUND_ACTIVE` (pydantic raises on the `OAUTH2_OUTBOUND_ACTIVE=True` kwarg because `init_forbid_extra` is set; the default test fails on the missing attribute).

- [ ] **Step 3: Add the field**

In `src/orchestrator_agent/settings.py`, inside `class AgentSettings`, immediately after the `AGENT_DEBUG` field (currently line 67), add:

```python
    OAUTH2_OUTBOUND_ACTIVE: bool | None = Field(
        default=None,
        description="Enable OAuth2 client-credentials auth on outgoing requests. When unset, follows OAUTH2_ACTIVE.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PATH="$HOME/.local/bin:$PATH" uv run pytest tests/test_settings.py -v`
Expected: PASS (all settings tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
PATH="$HOME/.local/bin:$PATH" git add src/orchestrator_agent/settings.py tests/test_settings.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Add OAUTH2_OUTBOUND_ACTIVE setting"
```

---

### Task 2: Resolve outbound auth from the new setting

**Files:**
- Modify: `src/orchestrator_agent/auth.py` (new import; rewrite `auth_enabled` property at lines 27-29)
- Test: `tests/test_auth.py` (add cases to `TestOAuthTokenManager`)

- [ ] **Step 1: Write the failing tests**

Append these methods inside the `TestOAuthTokenManager` class in `tests/test_auth.py`:

```python
    async def test_auth_enabled_follows_oauth_active_when_unset(self, monkeypatch):
        monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", None)
        mgr = OAuthTokenManager()

        monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE", True)
        assert mgr.auth_enabled is True

        monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE", False)
        assert mgr.auth_enabled is False

    async def test_outbound_true_overrides_inactive_incoming(self, monkeypatch, token_response):
        monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE", False)
        monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", True)
        monkeypatch.setattr(
            "orchestrator_agent.auth.oauth2lib_settings.OAUTH2_TOKEN_URL", "https://idp.example.com/token"
        )
        monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_RESOURCE_SERVER_ID", "test-client")
        monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_RESOURCE_SERVER_SECRET", "test-secret")
        mgr = OAuthTokenManager()
        assert mgr.auth_enabled is True

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = token_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("orchestrator_agent.auth.httpx.AsyncClient", return_value=mock_client):
            assert await mgr.get_token() == "tok-123"

    async def test_outbound_false_overrides_active_incoming(self, monkeypatch):
        monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE", True)
        monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)
        mgr = OAuthTokenManager()
        assert mgr.auth_enabled is False
        assert await mgr.get_token() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PATH="$HOME/.local/bin:$PATH" uv run pytest tests/test_auth.py -k "auth_enabled or overrides" -v`
Expected: FAIL/ERROR — `orchestrator_agent.auth` has no attribute `agent_settings` (the monkeypatch target does not resolve yet).

- [ ] **Step 3: Add the import and rewrite `auth_enabled`**

In `src/orchestrator_agent/auth.py`, add the import below the existing `oauth2lib_settings` import (line 16):

```python
from orchestrator_agent.settings import agent_settings
```

Replace the `auth_enabled` property (currently lines 27-29):

```python
    @property
    def auth_enabled(self) -> bool:
        return oauth2lib_settings.OAUTH2_ACTIVE
```

with:

```python
    @property
    def auth_enabled(self) -> bool:
        if agent_settings.OAUTH2_OUTBOUND_ACTIVE is None:
            return oauth2lib_settings.OAUTH2_ACTIVE
        return agent_settings.OAUTH2_OUTBOUND_ACTIVE
```

- [ ] **Step 4: Run the full auth + settings suites to verify they pass**

Run: `PATH="$HOME/.local/bin:$PATH" uv run pytest tests/test_auth.py tests/test_settings.py -v`
Expected: PASS (all existing auth tests still pass because the default is `None`, plus the three new ones).

- [ ] **Step 5: Type-check the changed modules**

Run: `PATH="$HOME/.local/bin:$PATH" uv run mypy src/orchestrator_agent/auth.py src/orchestrator_agent/settings.py`
Expected: no errors (no new import cycle; `settings.py` does not import `auth.py`).

- [ ] **Step 6: Commit**

```bash
PATH="$HOME/.local/bin:$PATH" git add src/orchestrator_agent/auth.py tests/test_auth.py
PATH="$HOME/.local/bin:$PATH" git commit -m "Resolve outbound OAuth from OAUTH2_OUTBOUND_ACTIVE"
```

---

### Task 3: Document the new toggle

**Files:**
- Modify: `README.md` (Configuration table)
- Modify: `.env.example` (best-effort; see note)

- [ ] **Step 1: Add the README config row**

In `README.md`, in the Configuration table, add this row immediately after the `OAUTH2_ACTIVE` row:

```markdown
| `OAUTH2_OUTBOUND_ACTIVE` | *(unset)* | Enable OAuth2 client-credentials auth on outgoing requests to orchestrator-core. When unset, follows `OAUTH2_ACTIVE`; set to `true`/`false` to control outbound auth independently of incoming auth |
```

- [ ] **Step 2: Add the `.env.example` entry (best-effort)**

Add the following line to `.env.example`, near the other `OAUTH2_*` entries:

```bash
# OAUTH2_OUTBOUND_ACTIVE=true  # outbound auth independent of OAUTH2_ACTIVE; unset = follow OAUTH2_ACTIVE
```

Note: `.env.example` may be under a tooling permission-denied path. If the edit cannot be made, leave it and tell the user to add the line manually — do not block the task on it.

- [ ] **Step 3: Verify the full test suite still passes**

Run: `PATH="$HOME/.local/bin:$PATH" uv run pytest`
Expected: PASS (all tests).

- [ ] **Step 4: Commit**

```bash
PATH="$HOME/.local/bin:$PATH" git add README.md .env.example
PATH="$HOME/.local/bin:$PATH" git commit -m "Document OAUTH2_OUTBOUND_ACTIVE toggle"
```

(If `.env.example` could not be modified, drop it from the `git add` and commit only `README.md`.)

---

## Notes for the implementer

- Do **not** add `Co-Authored-By` lines to commit messages.
- Prefix git/test/mypy commands with `PATH="$HOME/.local/bin:$PATH"` so pre-commit hooks can find `uv`.
- The incoming `AuthMiddleware` (`src/orchestrator_agent/security.py`) is intentionally untouched — it already gives "incoming off" via `OAUTH2_ACTIVE=False`.
- Outbound credentials (`OAUTH2_TOKEN_URL`, `OAUTH2_RESOURCE_SERVER_ID`, `OAUTH2_RESOURCE_SERVER_SECRET`) are still required when outbound auth is enabled; this plan does not change how they are read.
