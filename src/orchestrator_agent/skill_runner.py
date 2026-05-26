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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

import structlog
from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import FunctionToolCallEvent, PartDeltaEvent
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.events import RunContext, make_step_active_event
from orchestrator_agent.state import SearchState
from orchestrator_agent.utils import log_agent_request

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from orchestrator_agent.skills import Skill

logger = structlog.get_logger(__name__)


@dataclass
class SkillRunner:
    """Executes any Skill: creates pydantic-ai Agent, streams events.

    Concrete class parameterized by Skill — no subclasses needed.
    Plan advancement is handled by the agent loop, not here.
    """

    skill: Skill
    model: str | Model
    debug: bool = False
    _tool_calls_in_current_run: list[str] = field(default_factory=list, init=False, repr=False)
    _last_run_result: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = Agent(
            model=self.model,
            deps_type=StateDeps[SearchState],
            retries=2,
            toolsets=self.skill.toolsets,
        )

    async def run(self, ctx: RunContext, *, reasoning: str | None = None, planned: bool = True) -> AsyncIterator[Any]:
        """Execute the skill: stream events from the LLM agent.

        Args:
            ctx: Current run context
            reasoning: Optional reasoning from the execution plan task
            planned: Whether this task was created by the planner (vs direct invocation)
        """
        step_name = self.skill.name
        ctx.state.memory.start_step(step_name)
        yield make_step_active_event(step_name, reasoning)

        self._tool_calls_in_current_run = []
        self._last_run_result = None

        prompt = self.skill.get_prompt(ctx.state)
        # Use the planner's task reasoning as the user message so each skill
        # only sees its scoped task, not the full multi-part user request.
        override = reasoning if planned else None
        message_history = ctx.state.memory.get_message_history(
            max_turns=5, scope=self.skill.memory_scope, override_current_message=override
        )
        state_deps = StateDeps(ctx.state)

        if self.debug:
            log_agent_request(step_name, prompt, message_history)

        async for event in self._agent.run_stream_events(
            instructions=prompt,
            deps=state_deps,
            message_history=message_history,
        ):
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

        if self._last_run_result is None:
            logger.warning(f"{step_name}: No result captured")
