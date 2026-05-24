# Copyright 2019-2026 SURF, GÉANT.
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

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

import structlog
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import FunctionToolResultEvent, PartDeltaEvent, ToolReturn, ToolReturnPart
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.events import RunContext, make_step_active_event
from orchestrator_agent.state import SearchState
from orchestrator_agent.utils import log_agent_request

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from orchestrator_agent.skills import DelegationSkill, InternalSkill, Skill

logger = structlog.get_logger(__name__)


@dataclass
class SkillRunner:
    """Executes any Skill: streams events from the appropriate dispatch path.

    - :class:`InternalSkill` runs through a pydantic-ai Agent — LLM picks
      tool calls.
    - :class:`DelegationSkill` runs the skill's handler directly with the
      task reasoning as the query — no second LLM call.
    """

    skill: Skill
    model: str | Model
    debug: bool = False
    _tool_calls_in_current_run: list[str] = field(default_factory=list, init=False, repr=False)
    _last_run_result: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from orchestrator_agent.skills import InternalSkill

        # Annotated with the concrete generic so mypy resolves the deps_type /
        # run_stream_events overloads correctly. Without it the default
        # generic params are ``[None, str]`` and ``StateDeps[SearchState]``
        # mismatches ``type[None]``.
        self._agent: Agent[StateDeps[SearchState], str] | None
        if isinstance(self.skill, InternalSkill):
            self._agent = Agent(
                model=self.model,
                deps_type=StateDeps[SearchState],
                retries=2,
                toolsets=self.skill.toolsets,
            )
        else:
            self._agent = None

    async def run(self, ctx: RunContext, *, reasoning: str | None = None, planned: bool = True) -> AsyncIterator[Any]:
        from orchestrator_agent.skills import DelegationSkill

        step_name = self.skill.name
        ctx.state.memory.start_step(step_name)
        yield make_step_active_event(step_name, reasoning)

        self._tool_calls_in_current_run = []
        self._last_run_result = None

        if isinstance(self.skill, DelegationSkill):
            async for event in self._run_delegation(ctx, reasoning or ""):
                yield event
        else:
            async for event in self._run_internal(ctx, reasoning, planned):
                yield event

        if self._last_run_result is None:
            logger.warning(f"{step_name}: No result captured")

    async def _run_delegation(self, ctx: RunContext, query: str) -> AsyncIterator[Any]:
        """Direct dispatch: call the delegation handler with the planner's query."""

        skill: DelegationSkill = self.skill  # type: ignore[assignment]
        tool_call_id = str(uuid.uuid4())
        tool_name = f"ask_{getattr(skill, 'short_name', skill.name.lower().replace(' ', '_'))}"

        try:
            tool_return: ToolReturn = await skill.handler(ctx.state, query)
        except Exception as e:
            logger.error(f"{skill.name}: delegation failed", error=str(e), exc_info=True)
            raise

        # Emit a FunctionToolResultEvent so the A2A / AG-UI adapters see the
        # result on the same wire as LLM-driven tool calls. If a domain agent
        # ever starts emitting artifacts, ``tool_return.metadata`` carries them
        # through unchanged.
        yield FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name=tool_name,
                content=tool_return.return_value,
                tool_call_id=tool_call_id,
                metadata=tool_return.metadata,
            )
        )

        # Final result for the planner loop to consume as this task's answer.
        text = tool_return.return_value if isinstance(tool_return.return_value, str) else str(tool_return.return_value)
        result = AgentRunResult(output=text)
        self._last_run_result = result
        yield AgentRunResultEvent(result=result)

    async def _run_internal(self, ctx: RunContext, reasoning: str | None, planned: bool) -> AsyncIterator[Any]:
        """LLM-driven dispatch via pydantic-ai Agent."""
        from pydantic_ai.messages import FunctionToolCallEvent

        skill: InternalSkill = self.skill  # type: ignore[assignment]
        if self._agent is None:
            raise RuntimeError("InternalSkill runner reached _run_internal without a pydantic-ai Agent")
        step_name = skill.name

        prompt = skill.get_prompt(ctx.state)
        override = reasoning if planned else None
        message_history = ctx.state.memory.get_message_history(
            max_turns=5, scope=skill.memory_scope, override_current_message=override
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
