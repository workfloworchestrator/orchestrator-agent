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

from enum import Enum
from typing import Any, Literal
from uuid import UUID

from orchestrator.search.filters import FilterTree
from orchestrator.search.query.queries import Query
from pydantic import BaseModel, Field

from orchestrator_agent.memory import Memory


class TaskAction(str, Enum):
    """The action to perform for a task."""

    SEARCH = "search"
    AGGREGATION = "aggregation"
    RESULT_ACTIONS = "result_actions"
    TEXT_RESPONSE = "text_response"
    WORKFLOW_FORM_FILL = "workflow_form_fill"
    IMS_LOOKUP = "ims_lookup"
    INCIDENT_LOOKUP = "incident_lookup"
    JIRA_OPERATIONS = "jira_operations"
    TELEMETRY_QUERY = "telemetry_query"
    ALARM_QUERY = "alarm_query"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Executable task descriptor for routing to skills.

    ``reasoning`` doubles as the per-task instruction. For LLM-driven skills
    it's the prompt-level "what I'm doing"; for direct-dispatch delegation
    skills it's the focused English question forwarded to the domain agent
    via the skill's single tool.
    """

    action_type: TaskAction = Field(
        description="Which skill to execute. Pick the most specific delegated skill when the question fits one domain; pick a built-in skill (search / aggregation / result_actions / workflow_form_fill / text_response) when the work happens inside this agent."
    )
    reasoning: str = Field(
        description=(
            "For built-in skills: human-readable explanation of what this task accomplishes "
            "(e.g. 'Search for active subscriptions created in 2024'). "
            "For delegated skills: phrase this as a focused English question for the domain "
            "agent — it's forwarded verbatim. Include all IDs / timestamps / filters the "
            "domain agent needs to answer."
        )
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, exclude=True, description="Task execution status (managed internally)"
    )
    planned: bool = Field(
        default=True, exclude=True, description="Whether this task was created by the planner (vs direct invocation)"
    )


class ExecutionPlan(BaseModel):
    """Sequential execution plan — structured output from the Planner LLM."""

    tasks: list[Task] = Field(
        description='List of tasks to execute in order. Use multiple tasks for compound requests (e.g., "find X and export" needs 2 tasks).'
    )


class FinalAnswer(BaseModel):
    """Synthesized final answer.

    Structured output from the Planner LLM after
    task results have been folded back into its context.
    """

    answer: str = Field(description="Coherent, user-facing answer composed from the executed plan's results.")


FormFillState = Literal["gathering", "summary", "done"]


class FormFillSession(BaseModel):
    """Multi-turn state for a workflow form-fill session.

    Persisted on ``SearchState`` so the form can resume across A2A turns.
    """

    workflow_key: str
    page_inputs: list[dict[str, Any]] = Field(default_factory=list)
    current_page_schema: dict[str, Any] | None = None
    current_form_id: str | None = None
    state: FormFillState = "gathering"


class SearchState(BaseModel):
    """Agent state for search operations.

    Tracks the current search context and execution status.
    """

    user_input: str = ""  # Original user input text from current conversation turn
    run_id: UUID | None = None
    query_id: UUID | None = None  # ID of the last persisted query (set by run_search/run_aggregation)
    query: Query | None = None
    pending_filters: FilterTree | None = None
    memory: Memory = Field(default_factory=Memory)
    form_fill: FormFillSession | None = None
    adapter_metadata: dict[str, Any] | None = (
        None  # opaque per-turn metadata from the surface adapter (e.g. button-click envelopes); skills read their own keys out of it
    )

    class Config:
        arbitrary_types_allowed = True
