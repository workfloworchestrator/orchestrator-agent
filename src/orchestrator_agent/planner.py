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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

import structlog
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.events import (
    PlanCreatedTaskValue,
    RunContext,
    make_plan_created_event,
    make_step_active_event,
)
from orchestrator_agent.memory import MemoryScope, ToolStep
from orchestrator_agent.prompts import get_planning_prompt, get_synthesis_prompt
from orchestrator_agent.skill_runner import SkillRunner
from orchestrator_agent.state import ExecutionPlan, FinalAnswer, SearchState, Task, TaskAction, TaskStatus
from orchestrator_agent.utils import log_agent_request, log_execution_plan

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from orchestrator_agent.skills import Skill

logger = structlog.get_logger(__name__)


@dataclass
class Planner:
    """Creates and executes plans via LLM.

    Owns the full plan lifecycle: create plan, iterate tasks,
    run SkillRunners, handle replanning on failure.
    """

    model: str | Model
    skills: dict[TaskAction, Skill]
    debug: bool = False

    def __post_init__(self) -> None:
        # output_type is overridden per call: ExecutionPlan for the planning
        # phase, FinalAnswer for the synthesis phase.
        self._agent = Agent(
            model=self.model,
            deps_type=StateDeps[SearchState],
            output_type=ExecutionPlan,
            name="planner",
            retries=2,
        )

    async def _create_plan(self, ctx: RunContext) -> ExecutionPlan:
        """Create an execution plan via LLM."""
        ctx.state.memory.start_step("Planner")

        logger.info("Planner: Creating execution plan")

        message_history = ctx.state.memory.get_message_history(max_turns=5, scope=MemoryScope.FULL)
        prompt = get_planning_prompt(ctx.state)

        if self.debug:
            log_agent_request("Planner", prompt, message_history)

        result = await self._agent.run(instructions=prompt, message_history=message_history, deps=StateDeps(ctx.state))

        plan = result.output

        if self.debug:
            log_execution_plan(plan)

        logger.info(
            "Planner: Plan created",
            num_tasks=len(plan.tasks),
            tasks=[f"{i+1}. {t.reasoning}" for i, t in enumerate(plan.tasks)],
        )

        return plan

    async def _run_tasks(
        self, ctx: RunContext, tasks: list[Task], task_results: list[tuple[Task, str]]
    ) -> AsyncIterator[Any]:
        """Execute tasks sequentially, yielding events. Sets task.status on each.

        Appends ``(task, final_text)`` for each successful task to
        ``task_results`` so the synthesis pass can read them.
        """
        for task in tasks:
            skill = self.skills.get(task.action_type)
            if not skill:
                logger.warning(f"Unknown task type: {task.action_type}, skipping")
                continue

            if task.action_type in (TaskAction.SEARCH, TaskAction.AGGREGATION):
                ctx.state.query = None
                ctx.state.pending_filters = None

            task.status = TaskStatus.EXECUTING
            runner = SkillRunner(skill=skill, model=self.model, debug=self.debug)

            try:
                async for event in runner.run(ctx, reasoning=task.reasoning, planned=task.planned):
                    yield event
                task.status = TaskStatus.COMPLETED
                if runner._last_run_result is not None:
                    task_results.append((task, str(runner._last_run_result.output)))
            except Exception as e:
                logger.error(f"Task failed: {task.action_type}", error=str(e))
                task.status = TaskStatus.FAILED
                ctx.state.memory.record_tool_step(
                    ToolStep(
                        step_type="error", description=f"{skill.name} failed: {e}", success=False, error_message=str(e)
                    )
                )
                break

    async def _synthesize(
        self, ctx: RunContext, tasks: list[Task], task_results: list[tuple[Task, str]]
    ) -> AsyncIterator[Any]:
        """Synthesis pass: produce a final answer from the executed plan's results.

        Reuses the planner agent with ``FinalAnswer`` output. Skipped on
        single-task plans — the task's own result (artifact or text) is the
        answer, and adding a synthesizer reply on top would be redundant.
        """
        if len(tasks) <= 1:
            return

        ctx.state.memory.start_step("Synthesizer")
        yield make_step_active_event("Synthesizer")

        prompt = get_synthesis_prompt(ctx.state, tasks, task_results)
        message_history = ctx.state.memory.get_message_history(max_turns=5, scope=MemoryScope.FULL)

        if self.debug:
            log_agent_request("Synthesizer", prompt, message_history)

        result = await self._agent.run(
            instructions=prompt,
            message_history=message_history,
            deps=StateDeps(ctx.state),
            output_type=FinalAnswer,
        )

        answer = result.output.answer if isinstance(result.output, FinalAnswer) else str(result.output)
        yield AgentRunResultEvent(result=AgentRunResult(output=answer))

    async def execute(self, ctx: RunContext, *, target_action: TaskAction | None = None) -> AsyncIterator[Any]:
        """Create and execute a plan, then synthesise a final answer if needed.

        Args:
            ctx: Current run context
            target_action: If set, skip planning and execute this single action directly.
                No synthesis runs in this path — the surface explicitly named the skill,
                so its result is the answer.
        """
        if target_action:
            tasks = [Task(action_type=target_action, reasoning="", planned=False)]
            task_results: list[tuple[Task, str]] = []
            async for event in self._run_tasks(ctx, tasks, task_results):
                yield event
            return

        yield make_step_active_event("Planner")
        tasks = (await self._create_plan(ctx)).tasks

        plan_tasks = [
            PlanCreatedTaskValue(skill_name=skill.name, reasoning=task.reasoning)
            for task in tasks
            if (skill := self.skills.get(task.action_type))
        ]
        yield make_plan_created_event(plan_tasks)

        task_results = []
        async for event in self._run_tasks(ctx, tasks, task_results):
            yield event

        async for event in self._synthesize(ctx, tasks, task_results):
            yield event
