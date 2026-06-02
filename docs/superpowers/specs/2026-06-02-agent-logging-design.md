# Prompt/response DEBUG logging and agent-event TRACE logging

**Date:** 2026-06-02
**Status:** Approved

## Problem

We want full observability of the LLM agent's I/O:

- **All prompts** sent to the agent (the Planner and each Skill) logged at **DEBUG**.
- **All responses** from the agent logged at **DEBUG**.
- **All agent events** (the pydantic-ai stream events) logged at **TRACE**.

Today:

- Prompts are partially covered: `log_agent_request` (`src/orchestrator_agent/utils.py:52`) logs instructions + message_history at DEBUG, but it is gated behind the `AGENT_DEBUG` flag in `planner.py:73` and `skill_runner.py:83`, so setting the log level alone does not surface them.
- Responses are not logged (only a message-count line).
- Stream events are logged at DEBUG (with token deltas skipped), not TRACE.
- There is no TRACE level: structlog here routes through stdlib logging (`structlog.stdlib.BoundLogger`, configured by `nwastdlib.logging.initialise_logging`), and the standalone agent app (`src/orchestrator_agent/app.py`) never calls `initialise_logging` at all.

## Goal

Make prompts and responses appear at DEBUG and all (structural) agent events appear at TRACE, controlled purely by `LOG_LEVEL`, with a real TRACE level wired into the existing stdlib-routed structlog stack.

## Decisions (from brainstorming)

- **Log setup:** add a logging-setup module that registers a TRACE level and initializes logging at app startup, driven by `LOG_LEVEL`.
- **Gating:** purely by log level. Drop the `AGENT_DEBUG` gate and remove the now-dead `debug` plumbing.
- **Event scope:** structural events only — log every stream event at TRACE *except* per-token `PartDeltaEvent`.

## Verified mechanism

structlog's stdlib `BoundLogger` does not know level 5: `logger.log(5, ...)` raises `KeyError 5` and `logger.trace(...)` raises `AttributeError`. The working approach (confirmed empirically against this venv) is to add a `trace` method to **both** the stdlib `Logger` and structlog's `BoundLogger`:

```python
import logging
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def _logger_trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = _logger_trace  # type: ignore[attr-defined]

import structlog
def _bound_trace(self, event=None, *args, **kw):
    return self._proxy_to_logger("trace", event, *args, **kw)

structlog.stdlib.BoundLogger.trace = _bound_trace  # type: ignore[attr-defined]
```

Confirmed: with `LOG_LEVEL=TRACE`, `logger.trace("e", k=v)` renders a `trace` line; with `LOG_LEVEL=DEBUG` it is filtered out. `structlog.get_logger` is typed `-> Any`, so `logger.trace(...)` call sites pass mypy strict with no ignores; only the two monkeypatch assignments need `# type: ignore[attr-defined]`.

## Design

### 1. New module `src/orchestrator_agent/logging_config.py`

- `TRACE_LEVEL: int = 5`.
- `register_trace_level() -> None`: idempotent. Runs `logging.addLevelName(TRACE_LEVEL, "TRACE")`, attaches `trace` to `logging.Logger` (guarded by `isEnabledFor`), and attaches the proxy `trace` to `structlog.stdlib.BoundLogger`. Split out from `setup_logging` so it is testable without performing global logging configuration.
- `setup_logging() -> None`: calls `register_trace_level()`, then imports and calls `nwastdlib.logging.initialise_logging()` (imported inside the function so `LOG_LEVEL` is read at startup, not at module import). Idempotent / safe to call once.
- The module's responsibility: own the TRACE level and the agent's logging initialization. One clear purpose.

### 2. Call `setup_logging()` at app startup

In `src/orchestrator_agent/app.py`, call `setup_logging()` as the first statement in the `lifespan` context manager (before `init_database`/adapter startup). The standalone agent currently configures no logging, so this also fixes DEBUG-level visibility in general. `LOG_LEVEL=TRACE` → everything; `DEBUG` → prompts + responses; `INFO` → normal.

### 3. Prompts at DEBUG — un-gate

- Remove `if self.debug:` around `log_agent_request(...)` in `planner.py:73-74` and `skill_runner.py:83-84`; call it unconditionally. `log_agent_request` already logs at DEBUG (`utils.py:52`), so `LOG_LEVEL=DEBUG` surfaces every prompt (instructions + message_history) for the Planner and every Skill.
- Remove the `if self.debug:` around `log_execution_plan(plan)` in `planner.py:80-81`; call it unconditionally (it already logs at DEBUG).

### 4. Responses at DEBUG — new helper

Add to `src/orchestrator_agent/utils.py`:

```python
def log_agent_response(node_name: str, output: object) -> None:
    logger.debug("llm_response", node=node_name, output=output)
```

Call it:

- `planner.py`, after `result = await self._agent.run(...)`: `log_agent_response("Planner", plan)` (the `ExecutionPlan` output).
- `skill_runner.py`, in the `case AgentRunResultEvent():` branch: `log_agent_response(step_name, event.result.output)` (the skill's final text answer). This replaces the existing message-count `logger.debug` line.

### 5. Agent events at TRACE — structural only

Rework the `match event:` block in `skill_runner.py` (the `async for event in self._agent.run_stream_events(...)` loop) so it logs at TRACE, excluding token deltas:

```python
match event:
    case AgentRunResultEvent():
        self._last_run_result = event.result
        log_agent_response(step_name, event.result.output)
        logger.trace("agent_event", step=step_name, event_type="AgentRunResultEvent")
    case PartDeltaEvent():
        pass  # per-token deltas excluded from TRACE
    case _:
        logger.trace("agent_event", step=step_name, event_type=type(event).__name__, event=repr(event))
```

The tool-call tracking (`if isinstance(event, FunctionToolCallEvent)`) and its `try/except` stay unchanged.

### 6. Remove the dead `AGENT_DEBUG` plumbing

With the gates gone, the `debug` flag is unused end-to-end. Remove:

- `AGENT_DEBUG` field in `src/orchestrator_agent/settings.py:67`.
- The two `debug=agent_settings.AGENT_DEBUG` arguments in `app.py:38,41`.
- The `debug` parameter/attribute in `AgentAdapter.__init__` (`agent.py:87`) and the `debug=self.debug` passed to `Planner` (`agent.py:125`).
- The `debug` field on `Planner` (`planner.py:53`) and `debug=self.debug` passed to `SkillRunner` (`planner.py:104`).
- The `debug` field on `SkillRunner` (`skill_runner.py:47`).

## Error handling & scope

- `register_trace_level()` and `setup_logging()` are idempotent and side-effect-only.
- `repr(event)` is cheap and total; no special guarding needed. The existing tool-call `try/except` is untouched.
- `LOG_LEVEL` (read by `nwastdlib`) is the single control; no new setting is added.
- Out of scope: changing log output format/renderer, per-logger overrides, logging in the adapters or the top-level `agent.py` stream (the SkillRunner loop is the chokepoint for pydantic-ai agent events), and structured redaction of prompt/response content.

## Tests

`pytest` in `tests/` (new `tests/test_logging_config.py`, plus additions to `tests/test_utils.py`):

1. `register_trace_level()` registers the level (`logging.getLevelName(5) == "TRACE"`) and yields a callable `trace` on a structlog logger.
2. A `trace` record is emitted at level 5 and is filtered out at DEBUG — assert via `caplog.at_level(...)` capturing the `TRACE_LEVEL` record's presence/absence. `register_trace_level()` is idempotent (calling twice is safe).
3. `log_agent_response("Planner", <output>)` emits a single DEBUG `llm_response` record carrying `node` and `output` (capture structlog output / `caplog`).
4. SkillRunner event handling: with `self._agent.run_stream_events` monkeypatched to an async generator yielding a `PartDeltaEvent`, a `FunctionToolCallEvent`, and an `AgentRunResultEvent`, assert that the `PartDeltaEvent` produces no `agent_event` trace record while the other two do (and the result is captured). Use `pytest.mark.parametrize` where cases differ only by event/expectation.

Existing tests that construct `AgentAdapter`/`Planner`/`SkillRunner` with `debug=` must be updated to drop that argument (section 6).

## Out of scope

- New log renderers, JSON vs colored output selection (already handled by `LOG_OUTPUT`).
- Redaction/truncation of large prompt or response payloads.
- Logging at the adapter layer or the top-level `agent.run_stream_events` custom events.
