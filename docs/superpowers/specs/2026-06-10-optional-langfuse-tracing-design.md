# Optional Langfuse Tracing — Design

**Date:** 2026-06-10
**Status:** Approved (pending spec review)

## Goal

Add optional [Langfuse](https://langfuse.com) tracing to the orchestrator-agent. Langfuse must be:

- **Opt-in at runtime** via an explicit `LANGFUSE_ENABLED` flag (default off).
- **Optional as a dependency** — excluded from the base install, available via a `langfuse` extra.
- **Fail-safe** — any misconfiguration (missing extra, missing/invalid keys, auth failure) degrades to "no tracing" and never prevents the app from starting.

Additionally, produce a **build artifact (container image) with Langfuse installed**, alongside the existing base image.

## Background

Langfuse v3 is OpenTelemetry-based. `langfuse.get_client()` reads credentials from the
environment and configures a global OTel `TracerProvider`. pydantic-ai emits spans to that
global provider once instrumentation is enabled via the `Agent.instrument_all()` classmethod.

In this codebase the real LLM calls happen inside the `Planner` and `SkillRunner` agents
(separate `Agent` instances), not in `AgentAdapter`. Because `instrument_all()` is global, a
single call instruments every agent regardless of construction order — no per-agent wiring.

## Design

### 1. Packaging — optional `langfuse` extra

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
langfuse = ["langfuse>=3.0.0"]
```

Base install (`uv sync`) excludes it. Opt in with `uv sync --extra langfuse` /
`pip install orchestrator-agent[langfuse]`. The lockfile is regenerated (`uv lock`) so the extra
resolves under `--frozen` in the Docker build.

### 2. Settings — single enable flag

Add one field to `AgentSettings` (`settings.py`):

```python
LANGFUSE_ENABLED: bool = Field(
    default=False,
    description="Enable Langfuse OpenTelemetry tracing. Requires the 'langfuse' extra and "
    "the LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST environment variables.",
)
```

Credentials and host are **not** added to `AgentSettings`; the langfuse SDK reads its own native
env vars (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) directly.

### 3. New module `observability.py`

Langfuse and `Agent` are imported lazily *inside* the functions so the module is importable
without the extra installed.

```python
def configure_langfuse() -> object | None:
    """Enable Langfuse OTel tracing if configured. Returns the client, or None if disabled/unavailable."""
    if not agent_settings.LANGFUSE_ENABLED:
        return None
    try:
        from langfuse import get_client
        from pydantic_ai import Agent
    except ImportError:
        logger.error("LANGFUSE_ENABLED is set but the 'langfuse' extra is not installed; tracing disabled")
        return None

    client = get_client()
    if not client.auth_check():
        logger.warning("Langfuse auth check failed; tracing disabled. Check LANGFUSE_* env vars")
        return None

    Agent.instrument_all()
    logger.info("Langfuse tracing enabled")
    return client


def shutdown_langfuse(client: object) -> None:
    """Flush pending spans to Langfuse on shutdown."""
    client.flush()  # type: ignore[attr-defined]
```

### 4. Wiring in `app.py` lifespan

Near the top of `lifespan`, before/around adapter creation:

```python
langfuse_client = configure_langfuse()
```

On shutdown (after adapters stop):

```python
if langfuse_client is not None:
    shutdown_langfuse(langfuse_client)
```

`instrument_all()` is global, so ordering relative to adapter construction is not critical.

### 5. Build artifact with Langfuse on

Parameterize the **existing** `Dockerfile` with a build arg (no duplicate Dockerfile):

```dockerfile
ARG UV_EXTRA_ARGS=""
RUN uv sync --frozen --no-dev ${UV_EXTRA_ARGS}
```

- Base image: built as today (`UV_EXTRA_ARGS` empty).
- Langfuse image: `docker build --build-arg UV_EXTRA_ARGS="--extra langfuse" .`.

CI (`.github/workflows/ci.yml`) `build` and `deploy` jobs gain a `variant` matrix
(`default`, `langfuse`) that maps to the build arg and a tag suffix:

- `default` → existing tags (e.g. `:latest`, `:sha-…`).
- `langfuse` → same tags with a `-langfuse` suffix (e.g. `:latest-langfuse`).

## Failure Behavior

| Condition | Result |
|-----------|--------|
| `LANGFUSE_ENABLED` false | `configure_langfuse()` returns `None`; langfuse never imported |
| Enabled, extra not installed | Log error, return `None`, app continues without tracing |
| Enabled, missing/invalid keys (`auth_check` fails) | Log warning, return `None`, app continues |
| Enabled, all good | `Agent.instrument_all()` called, spans exported, flushed on shutdown |

The app never fails to start because of Langfuse.

## Testing

Unit tests (`tests/`), langfuse mocked/monkeypatched (it is not a dev dependency):

1. **Disabled** → `configure_langfuse()` returns `None` and does not import langfuse.
2. **Enabled, import fails** → returns `None`, error logged.
3. **Enabled, `auth_check` passes** → `Agent.instrument_all` invoked, client returned.
4. **Enabled, `auth_check` fails** → returns `None`, warning logged.
5. **`shutdown_langfuse`** → calls `client.flush()`.

## Out of Scope

- Custom spans / manual scoring beyond pydantic-ai's automatic instrumentation.
- Adding Langfuse credentials to `AgentSettings`.
- Per-adapter (AG-UI / A2A / MCP) tracing differences — global instrumentation covers all.

## Files Touched

- `pyproject.toml` — `[project.optional-dependencies]` langfuse extra.
- `uv.lock` — regenerated.
- `src/orchestrator_agent/settings.py` — `LANGFUSE_ENABLED` field.
- `src/orchestrator_agent/observability.py` — new module.
- `src/orchestrator_agent/app.py` — lifespan wiring.
- `Dockerfile` — `UV_EXTRA_ARGS` build arg.
- `.github/workflows/ci.yml` — `variant` matrix for build/deploy.
- `tests/test_observability.py` — new tests.
