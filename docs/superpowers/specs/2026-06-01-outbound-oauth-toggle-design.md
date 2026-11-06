# Decouple outbound OAuth from inbound OAuth

**Date:** 2026-06-01
**Status:** Approved

## Problem

A single flag, `oauth2lib_settings.OAUTH2_ACTIVE`, currently controls both directions of OAuth:

- **Incoming** request authentication: `AuthMiddleware` (`src/orchestrator_agent/security.py:121`) passes requests through unauthenticated when `OAUTH2_ACTIVE` is `False`.
- **Outgoing** request authentication: `OAuthTokenManager.auth_enabled` (`src/orchestrator_agent/auth.py:27-29`) also returns `OAUTH2_ACTIVE`. When `False`, `get_token()` returns `None` and calls to the orchestrator-core API go unauthenticated.

These cannot be configured independently. We need to support deployments where **incoming auth is off but outgoing auth is on** (e.g. the agent sits behind a trusted gateway that already authenticates callers, yet it must present a client-credentials token when calling orchestrator-core).

## Goal

Make inbound and outbound OAuth fully independent, while keeping every existing deployment's behavior unchanged with zero config changes.

## Design

### New setting

Add a tri-state field to `AgentSettings` in `src/orchestrator_agent/settings.py`:

```python
OAUTH2_OUTBOUND_ACTIVE: bool | None = Field(
    default=None,
    description="Enable OAuth2 client-credentials auth on outgoing requests. When unset, follows OAUTH2_ACTIVE.",
)
```

- `None` (default) = "follow `OAUTH2_ACTIVE`" — preserves current behavior for all existing deployments.
- `True` / `False` = explicit override, independent of `OAUTH2_ACTIVE`.

### Resolution

Change `OAuthTokenManager.auth_enabled` in `src/orchestrator_agent/auth.py` to resolve the tri-state at read-time:

```python
from orchestrator_agent.settings import agent_settings  # new import

@property
def auth_enabled(self) -> bool:
    if agent_settings.OAUTH2_OUTBOUND_ACTIVE is None:
        return oauth2lib_settings.OAUTH2_ACTIVE
    return agent_settings.OAUTH2_OUTBOUND_ACTIVE
```

The import does not create a cycle: `settings.py` does not import `auth.py`. `result_actions.py` already imports both modules.

### Why tri-state over two plain booleans

A plain `bool` defaulting to `False` would require a validator that copies `OAUTH2_ACTIVE` into it at construction, which freezes the value at startup and hides the coupling inside a validator. The `None` sentinel keeps resolution at read-time and in one obvious place, and makes "unset = follow the incoming flag" explicit.

### Resulting behavior

| `OAUTH2_ACTIVE` (incoming) | `OAUTH2_OUTBOUND_ACTIVE` | Incoming auth | Outbound auth |
|---|---|---|---|
| True | unset (`None`) | ON | ON (follows) |
| False | unset (`None`) | OFF | OFF (follows) |
| False | True | OFF | ON (target scenario) |
| True | False | ON | OFF |
| True | True | ON | ON |
| False | False | OFF | OFF |

### Unchanged components

- `security.py` (`AuthMiddleware`) — inbound gate already reads only `OAUTH2_ACTIVE`.
- `result_actions.py` — outbound calls already gate on `token_manager.auth_enabled` (including the 401/403 refresh-and-retry path).
- `app.py` — no change.
- Outbound client-credentials inputs (`OAUTH2_TOKEN_URL`, `OAUTH2_RESOURCE_SERVER_ID`, `OAUTH2_RESOURCE_SERVER_SECRET`) already live in `oauth2lib_settings` and are independent of `OAUTH2_ACTIVE`. They remain required when outbound auth is enabled.

## Tests

Add cases to `tests/test_auth.py` covering `auth_enabled` resolution (monkeypatching `orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE` and `orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE`):

1. `OUTBOUND_ACTIVE = None` follows `OAUTH2_ACTIVE` — both `True` and `False` cases.
2. `OUTBOUND_ACTIVE = True` with `OAUTH2_ACTIVE = False` → enabled; `get_token()` performs the fetch.
3. `OUTBOUND_ACTIVE = False` with `OAUTH2_ACTIVE = True` → disabled; `get_token()` returns `None`.

Existing tests pass unchanged because the default is `None` (no `OAUTH2_OUTBOUND_ACTIVE` env var in the test environment).

## Documentation

- Add an `OAUTH2_OUTBOUND_ACTIVE` row to the README Configuration table, noting it follows `OAUTH2_ACTIVE` when unset.
- Add a commented `OAUTH2_OUTBOUND_ACTIVE` line to `.env.example` if writable. (The file is under a permission-denied path; if the edit cannot be made automatically, the user adds it manually.)

## Out of scope

- Renaming `OAUTH2_ACTIVE` (would break existing deployments).
- Per-destination outbound auth policies (only orchestrator-core is called today).
- Token expiry/refresh-by-clock changes (current refresh-on-401/403 behavior is unchanged).
