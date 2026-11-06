# Agent Prompt/Response/Event Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log all agent prompts and responses at DEBUG and all structural agent stream-events at a new TRACE level, controlled purely by `LOG_LEVEL`, and remove the now-dead `AGENT_DEBUG` flag.

**Architecture:** A new `logging_config.py` registers a TRACE level (5) into the existing stdlib-routed structlog stack (adding a `trace` method to both `logging.Logger` and `structlog.stdlib.BoundLogger`) and initializes logging at app startup. Prompt logging is un-gated, a `log_agent_response` helper is added, and the SkillRunner stream loop logs every event at TRACE except per-token deltas. The unused `debug`/`AGENT_DEBUG` plumbing is then deleted.

**Tech Stack:** Python 3.11–3.13, `uv`, structlog (via `nwastdlib.logging`, stdlib `BoundLogger`), pydantic-ai, FastAPI, pytest. Lint/format: `ruff check` + `ruff format` (line-length 120). Types: `mypy` strict on `src/` (`structlog.get_logger` is typed `-> Any`, so `logger.trace(...)` call sites need no ignores).

**Project Python style (apply throughout):** comprehensions/itertools over imperative loops; no `break`/`continue` (early return); `match/case` over `isinstance` chains; tests use `@pytest.mark.parametrize` instead of near-duplicate functions.

**Commit convention:** no `Co-Authored-By` lines; prefix git commands with `export PATH="$HOME/.local/bin:$PATH"`. This repo formats with `ruff format` (NOT `black`). If a pre-commit hook fails, fix the cause and make a NEW commit (never `--amend`/`--no-verify`).

---

## File Structure

- **Create** `src/orchestrator_agent/logging_config.py` — owns the TRACE level and the agent's logging initialization.
- **Create** `tests/test_logging_config.py` — tests the TRACE level registration, emission, and filtering.
- **Create** `tests/test_skill_runner.py` — tests the event-trace classification helper.
- **Modify** `src/orchestrator_agent/app.py` — call `setup_logging()` at startup.
- **Modify** `src/orchestrator_agent/utils.py` + `tests/test_utils.py` — add `log_agent_response`.
- **Modify** `src/orchestrator_agent/skill_runner.py` — response (DEBUG) + events (TRACE) + un-gate prompt.
- **Modify** `src/orchestrator_agent/planner.py` — un-gate prompt/plan logging + add response.
- **Modify** `settings.py`, `app.py`, `agent.py`, `planner.py`, `skill_runner.py` — remove dead `debug`/`AGENT_DEBUG`.

---

## Task 1: TRACE level + logging setup module

**Files:**
- Create: `src/orchestrator_agent/logging_config.py`
- Create: `tests/test_logging_config.py`
- Modify: `src/orchestrator_agent/app.py` (call `setup_logging()` in `lifespan`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging_config.py`:

```python
"""Tests for the TRACE log level and logging setup."""

from __future__ import annotations

import logging
import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import structlog

from orchestrator_agent.logging_config import TRACE_LEVEL, register_trace_level


class TestRegisterTraceLevel:
    def test_registers_level_name_and_methods(self):
        register_trace_level()
        assert TRACE_LEVEL == 5
        assert logging.getLevelName(TRACE_LEVEL) == "TRACE"
        assert callable(getattr(logging.Logger, "trace", None))
        assert callable(getattr(structlog.stdlib.BoundLogger, "trace", None))

    def test_idempotent(self):
        register_trace_level()
        register_trace_level()
        assert logging.getLevelName(TRACE_LEVEL) == "TRACE"

    def test_trace_emits_at_trace_level(self, caplog):
        register_trace_level()
        log = logging.getLogger("test_trace_emit")
        with caplog.at_level(TRACE_LEVEL, logger="test_trace_emit"):
            log.trace("hello_trace")
        assert any(r.levelno == TRACE_LEVEL and "hello_trace" in r.getMessage() for r in caplog.records)

    def test_trace_filtered_below_debug(self, caplog):
        register_trace_level()
        log = logging.getLogger("test_trace_filter")
        with caplog.at_level(logging.DEBUG, logger="test_trace_filter"):
            log.trace("hidden_trace")
            log.debug("shown_debug")
        messages = [r.getMessage() for r in caplog.records]
        assert "shown_debug" in messages
        assert "hidden_trace" not in messages
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator_agent.logging_config'`.

- [ ] **Step 3: Create the module**

Create `src/orchestrator_agent/logging_config.py`:

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

"""TRACE log level registration and logging initialization for the agent."""

from __future__ import annotations

import logging
from typing import Any

import structlog

TRACE_LEVEL = 5


def _logger_trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


def _bound_logger_trace(self: structlog.stdlib.BoundLogger, event: str | None = None, *args: Any, **kw: Any) -> Any:
    return self._proxy_to_logger("trace", event, *args, **kw)


def register_trace_level() -> None:
    """Register a TRACE level (5) on stdlib logging and structlog. Idempotent."""
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    logging.Logger.trace = _logger_trace  # type: ignore[attr-defined]
    structlog.stdlib.BoundLogger.trace = _bound_logger_trace  # type: ignore[attr-defined]


def setup_logging() -> None:
    """Register the TRACE level and initialize structlog logging from LOG_LEVEL."""
    register_trace_level()
    # Imported here so LOG_LEVEL is read at startup, not at module import time.
    from nwastdlib.logging import initialise_logging

    initialise_logging()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Call `setup_logging()` at app startup**

In `src/orchestrator_agent/app.py`, add the import next to the other `orchestrator_agent` imports (after line 27 `from orchestrator_agent.settings import agent_settings`):

```python
from orchestrator_agent.logging_config import setup_logging
```

Then make `setup_logging()` the first statement in `lifespan`. The current start is:

```python
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: DB init, migration, adapter startup/shutdown."""
    init_database(agent_settings)  # type: ignore[arg-type]  # AgentSettings has DATABASE_URI which is all init_database needs
```

Change it to:

```python
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: DB init, migration, adapter startup/shutdown."""
    setup_logging()
    init_database(agent_settings)  # type: ignore[arg-type]  # AgentSettings has DATABASE_URI which is all init_database needs
```

- [ ] **Step 6: Verify and commit**

Run: `uv run pytest tests/test_logging_config.py -q && uv run mypy src/ && uv run ruff check . && uv run ruff format --check .`
Expected: tests pass, mypy clean, ruff clean, format clean.

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/logging_config.py tests/test_logging_config.py src/orchestrator_agent/app.py
git commit -m "Add TRACE log level and initialize logging at startup"
```

---

## Task 2: `log_agent_response` helper

**Files:**
- Modify: `src/orchestrator_agent/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils.py`. First extend the import at the top (currently `from orchestrator_agent.utils import current_timestamp_ms, log_agent_request, log_execution_plan`) to also import `log_agent_response`:

```python
from orchestrator_agent.utils import (
    current_timestamp_ms,
    log_agent_request,
    log_agent_response,
    log_execution_plan,
)
```

Then add this class at the end of the file:

```python
class TestLogAgentResponse:
    def test_logs_output(self, capsys):
        log_agent_response("Planner", {"plan": "value"})
        captured = capsys.readouterr()
        assert "llm_response" in captured.out
        assert "node=Planner" in captured.out

    def test_logs_string_output(self, capsys):
        log_agent_response("Search", "here are your results")
        captured = capsys.readouterr()
        assert "llm_response" in captured.out
        assert "here are your results" in captured.out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL — `ImportError: cannot import name 'log_agent_response'`.

- [ ] **Step 3: Add the helper**

In `src/orchestrator_agent/utils.py`, append after `log_agent_request`:

```python
def log_agent_response(node_name: str, output: object) -> None:
    logger.debug("llm_response", node=node_name, output=output)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_utils.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/utils.py tests/test_utils.py
git commit -m "Add log_agent_response helper for DEBUG response logging"
```

---

## Task 3: SkillRunner — responses at DEBUG, events at TRACE, un-gate prompt

**Files:**
- Modify: `src/orchestrator_agent/skill_runner.py`
- Create: `tests/test_skill_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_skill_runner.py`:

```python
"""Tests for SkillRunner event-trace classification."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

from orchestrator_agent.skill_runner import _event_trace_fields


class TestEventTraceFields:
    def test_skips_part_delta(self):
        event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="x"))
        assert _event_trace_fields(event, "Search") is None

    def test_logs_non_delta_event(self):
        fields = _event_trace_fields(object(), "Search")
        assert fields is not None
        assert fields["step"] == "Search"
        assert fields["event_type"] == "object"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_skill_runner.py -v`
Expected: FAIL — `ImportError: cannot import name '_event_trace_fields'`.

- [ ] **Step 3: Add the helper and update imports**

In `src/orchestrator_agent/skill_runner.py`, change the utils import (currently `from orchestrator_agent.utils import log_agent_request`) to:

```python
from orchestrator_agent.utils import log_agent_request, log_agent_response
```

Add the helper just after `logger = structlog.get_logger(__name__)` (line 34):

```python
def _event_trace_fields(event: Any, step_name: str) -> dict[str, str] | None:
    """Trace fields for a stream event, or None to skip per-token deltas."""
    match event:
        case PartDeltaEvent():
            return None
        case _:
            return {"step": step_name, "event_type": type(event).__name__, "event": repr(event)}
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `uv run pytest tests/test_skill_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Un-gate the prompt log and rework the event loop**

In `skill_runner.py`, remove the `if self.debug:` gate. The current lines are:

```python
        if self.debug:
            log_agent_request(step_name, prompt, message_history)
```

Replace with:

```python
        log_agent_request(step_name, prompt, message_history)
```

Then replace the event-handling block. The current block is:

```python
            try:
                if isinstance(event, FunctionToolCallEvent):
                    self._tool_calls_in_current_run.append(event.part.tool_name)
            except Exception as e:
                logger.error(f"Error tracking tool call: {e}", exc_info=True)

            match event:
                case AgentRunResultEvent():
                    self._last_run_result = event.result
                    logger.debug(
                        f"{step_name}: Captured final result with {len(event.result.new_messages())} new messages"
                    )
                case PartDeltaEvent():
                    pass
                case _:
                    logger.debug(f"{step_name}: Yielding event", event_type=type(event).__name__)

            yield event
```

Replace with:

```python
            try:
                if isinstance(event, FunctionToolCallEvent):
                    self._tool_calls_in_current_run.append(event.part.tool_name)
            except Exception as e:
                logger.error(f"Error tracking tool call: {e}", exc_info=True)

            match event:
                case AgentRunResultEvent():
                    self._last_run_result = event.result
                    log_agent_response(step_name, event.result.output)
                case _:
                    pass

            if (fields := _event_trace_fields(event, step_name)) is not None:
                logger.trace("agent_event", **fields)

            yield event
```

(`PartDeltaEvent` is still imported and used by `_event_trace_fields`; `AgentRunResultEvent` and `FunctionToolCallEvent` remain imported and used.)

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all pass (no test executes `SkillRunner.run`, so the `logger.trace` call path is not exercised in tests; the helper is covered directly).

- [ ] **Step 7: Verify types/lint and commit**

Run: `uv run mypy src/ && uv run ruff check . && uv run ruff format --check .`
Expected: clean.

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/skill_runner.py tests/test_skill_runner.py
git commit -m "Log skill responses at DEBUG and stream events at TRACE"
```

---

## Task 4: Planner — un-gate prompt/plan logging and add response

**Files:**
- Modify: `src/orchestrator_agent/planner.py`

- [ ] **Step 1: Update imports**

In `src/orchestrator_agent/planner.py`, change the utils import (currently `from orchestrator_agent.utils import log_agent_request, log_execution_plan`) to:

```python
from orchestrator_agent.utils import log_agent_request, log_agent_response, log_execution_plan
```

- [ ] **Step 2: Un-gate the request log**

The current lines are:

```python
        if self.debug:
            log_agent_request("Planner", prompt, message_history)

        result = await self._agent.run(instructions=prompt, message_history=message_history, deps=StateDeps(ctx.state))

        plan = result.output

        if self.debug:
            log_execution_plan(plan)
```

Replace with:

```python
        log_agent_request("Planner", prompt, message_history)

        result = await self._agent.run(instructions=prompt, message_history=message_history, deps=StateDeps(ctx.state))

        plan = result.output

        log_agent_response("Planner", plan)
        log_execution_plan(plan)
```

- [ ] **Step 3: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all pass (no test runs `Planner._create_plan`).

- [ ] **Step 4: Verify types/lint and commit**

Run: `uv run mypy src/ && uv run ruff check . && uv run ruff format --check .`
Expected: clean (note: `self.debug` is now unused in `planner.py` but still defined — removed in Task 5).

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/planner.py
git commit -m "Log planner prompt, response, and plan at DEBUG unconditionally"
```

---

## Task 5: Remove the dead `AGENT_DEBUG` / `debug` plumbing

**Files:**
- Modify: `src/orchestrator_agent/settings.py`
- Modify: `src/orchestrator_agent/app.py`
- Modify: `src/orchestrator_agent/agent.py`
- Modify: `src/orchestrator_agent/planner.py`
- Modify: `src/orchestrator_agent/skill_runner.py`

- [ ] **Step 1: Remove the setting**

In `src/orchestrator_agent/settings.py`, delete the line:

```python
    AGENT_DEBUG: bool = Field(default=False, description="Enable debug logging for agent execution")
```

- [ ] **Step 2: Remove `debug=` args in app.py**

In `src/orchestrator_agent/app.py`, the two lines:

```python
    a2a = A2AAdapter(AgentAdapter(agent_settings.create_model(), debug=agent_settings.AGENT_DEBUG), url=a2a_url)
```
```python
    mcp_app = MCPApp(AgentAdapter(agent_settings.create_model(), debug=agent_settings.AGENT_DEBUG))
```

become:

```python
    a2a = A2AAdapter(AgentAdapter(agent_settings.create_model()), url=a2a_url)
```
```python
    mcp_app = MCPApp(AgentAdapter(agent_settings.create_model()))
```

- [ ] **Step 3: Remove `debug` from AgentAdapter (`agent.py`)**

Delete the `debug: bool = False,` parameter from `AgentAdapter.__init__` (line 74), delete `self.debug = debug` (line 87), and change the Planner construction (line 125) from:

```python
            planner = Planner(model=self._model_ref, skills=self.skills, debug=self.debug)
```

to:

```python
            planner = Planner(model=self._model_ref, skills=self.skills)
```

- [ ] **Step 4: Remove `debug` from Planner (`planner.py`)**

Delete the `debug: bool = False` field (line 53) from the `Planner` dataclass, and change the SkillRunner construction (line 104) from:

```python
            runner = SkillRunner(skill=skill, model=self.model, debug=self.debug)
```

to:

```python
            runner = SkillRunner(skill=skill, model=self.model)
```

- [ ] **Step 5: Remove `debug` from SkillRunner (`skill_runner.py`)**

Delete the `debug: bool = False` field from the `SkillRunner` dataclass (line 47).

- [ ] **Step 6: Verify nothing references the removed names**

Run: `export PATH="$HOME/.local/bin:$PATH" && grep -rn "AGENT_DEBUG\|self.debug\|debug=" src/ tests/`
Expected: no matches (empty output).

- [ ] **Step 7: Run full suite, types, lint**

Run: `uv run pytest -q && uv run mypy src/ && uv run ruff check . && uv run ruff format --check .`
Expected: all pass / clean. (mypy strict will flag any leftover reference to the removed attributes.)

- [ ] **Step 8: Commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
git add src/orchestrator_agent/settings.py src/orchestrator_agent/app.py src/orchestrator_agent/agent.py src/orchestrator_agent/planner.py src/orchestrator_agent/skill_runner.py
git commit -m "Remove dead AGENT_DEBUG flag now that logging is level-driven"
```

---

## Task 6: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass (existing count + 8 new tests: 4 logging_config, 2 utils, 2 skill_runner).

- [ ] **Step 2: Types, lint, format**

Run: `uv run mypy src/ && uv run ruff check . && uv run ruff format --check .`
Expected: `Success`, `All checks passed!`, `N files already formatted`.

- [ ] **Step 3: Manual smoke check of the TRACE wiring (optional)**

Run:
```bash
export PATH="$HOME/.local/bin:$PATH"
LOG_LEVEL=TRACE uv run python -c "
from orchestrator_agent.logging_config import setup_logging
import structlog
setup_logging()
log = structlog.get_logger('smoke')
log.trace('trace_visible', k=1)
log.debug('debug_visible')
"
```
Expected: both a `trace` line and a `debug` line render. Re-running with `LOG_LEVEL=DEBUG` should show only the `debug` line.

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** Task 1 → spec §1, §2 (module + app startup) and the verified TRACE mechanism. Task 2 → spec §4 helper. Task 3 → spec §4 (skill response) + §5 (events at TRACE, deltas skipped) + §3 (un-gate skill prompt). Task 4 → spec §3 (un-gate planner prompt/plan) + §4 (planner response). Task 5 → spec §6 (remove `AGENT_DEBUG`). Task 6 → verification.
- **Why `_event_trace_fields` is a pure helper:** it makes the "skip token deltas, log everything else" rule unit-testable without configuring global structlog or running an LLM. The actual `logger.trace(...)` emission is verified by the Task 1 tests and the Task 6 smoke check.
- **Why the planner has no new unit test:** `Planner._create_plan` requires a live LLM; the change is un-gating + adding a `log_agent_response` call (both helpers are tested independently). Verification is "full suite stays green."
- **Lazy-logger ordering:** module-level `logger = structlog.get_logger(__name__)` returns a lazy proxy that resolves on first use. `setup_logging()` runs first in `lifespan`, before any agent log call, so the proxy resolves to the stdlib `BoundLogger` (which has `.trace`). No agent code logs before startup.
- **`debug` removal order:** additive logging (Tasks 1–4) lands first; `self.debug` becomes unused but still defined until Task 5 deletes it. Tests stay green throughout because no test constructs these objects with `debug=` (verified).
