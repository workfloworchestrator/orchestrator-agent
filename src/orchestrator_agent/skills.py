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

from dataclasses import dataclass
from typing import Any, Callable

from pydantic_ai.toolsets import AbstractToolset

from orchestrator_agent.memory import MemoryScope
from orchestrator_agent.prompts import (
    get_aggregation_execution_prompt,
    get_result_actions_prompt,
    get_search_execution_prompt,
    get_text_response_prompt,
    get_workflow_form_fill_prompt,
)
from orchestrator_agent.state import SearchState, TaskAction
from orchestrator_agent.tools import (
    aggregation_execution_toolset,
    aggregation_toolset,
    filter_building_toolset,
    result_actions_toolset,
    search_execution_toolset,
    workflow_forms_toolset,
)
from orchestrator_agent.tools.a2a_delegate import DelegateHandler, make_a2a_delegate_handler


@dataclass(frozen=True)
class Skill:
    """Base — metadata shared by every skill type.

    In the A2A protocol, one agent advertises many skills — discrete
    capabilities with metadata (name, description, tags). All Skill subtypes
    surface that metadata. The difference between subtypes is *how the
    skill is executed*:

    - :class:`InternalSkill` — the skill runs inside this agent: a pydantic-ai
      Agent is spun up with the skill's toolsets and prompt, the LLM picks
      tool calls.
    - :class:`DelegationSkill` — the skill forwards a single query to a
      remote domain agent over A2A. No LLM, no toolset, no prompt — the
      planner already chose this skill and produced the query.
    """

    action: TaskAction
    name: str
    description: str
    tags: list[str]
    memory_scope: MemoryScope


@dataclass(frozen=True)
class InternalSkill(Skill):
    """A skill executed inside this agent via an LLM tool-use loop."""

    toolsets: list[AbstractToolset[Any]]
    get_prompt: Callable[[SearchState], str]


@dataclass(frozen=True)
class DelegationSkill(Skill):
    """A skill that delegates to a domain agent via A2A.

    The handler receives ``(state, query)`` and performs the A2A round-trip.
    ``query`` is ``Task.reasoning`` produced by the planner — a focused
    English question phrased for the domain agent.
    """

    handler: DelegateHandler


SKILLS: dict[TaskAction, Skill] = {
    TaskAction.SEARCH: InternalSkill(
        action=TaskAction.SEARCH,
        name="Search",
        description="Find subscriptions, products, workflows, processes",
        tags=["search", "query"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        toolsets=[filter_building_toolset, search_execution_toolset],
        get_prompt=get_search_execution_prompt,
    ),
    TaskAction.AGGREGATION: InternalSkill(
        action=TaskAction.AGGREGATION,
        name="Aggregate",
        description="Count, sum, avg with grouping",
        tags=["aggregate", "analytics"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        toolsets=[filter_building_toolset, aggregation_toolset, aggregation_execution_toolset],
        get_prompt=get_aggregation_execution_prompt,
    ),
    TaskAction.RESULT_ACTIONS: InternalSkill(
        action=TaskAction.RESULT_ACTIONS,
        name="Result Actions",
        description="Export results or fetch entity details by ID (subscription, product, workflow)",
        tags=["export", "details", "lookup"],
        memory_scope=MemoryScope.MINIMAL,
        toolsets=[result_actions_toolset],
        get_prompt=get_result_actions_prompt,
    ),
    TaskAction.TEXT_RESPONSE: InternalSkill(
        action=TaskAction.TEXT_RESPONSE,
        name="Text Response",
        description="Answer general questions about the system",
        tags=["text", "help"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        toolsets=[],
        get_prompt=get_text_response_prompt,
    ),
    TaskAction.WORKFLOW_FORM_FILL: InternalSkill(
        action=TaskAction.WORKFLOW_FORM_FILL,
        name="Workflow Form Fill",
        description="Walk a user through a multi-page workflow form and start the workflow",
        tags=["workflow", "form"],
        memory_scope=MemoryScope.FULL,
        toolsets=[workflow_forms_toolset],
        get_prompt=get_workflow_form_fill_prompt,
    ),
    TaskAction.IMS_LOOKUP: DelegationSkill(
        action=TaskAction.IMS_LOOKUP,
        name="IMS Lookup",
        description="Look up IMS nodes, services or planned works",
        tags=["ims", "inventory", "delegate"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        handler=make_a2a_delegate_handler(short_name="ims", url_attr="IMS_AGENT_A2A_URL"),
    ),
    TaskAction.INCIDENT_LOOKUP: DelegationSkill(
        action=TaskAction.INCIDENT_LOOKUP,
        name="Incident Lookup",
        description="Look up or create customer incident tickets (CIM)",
        tags=["cim", "incident", "ticket", "delegate"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        handler=make_a2a_delegate_handler(short_name="cim", url_attr="CIM_AGENT_A2A_URL"),
    ),
    TaskAction.JIRA_OPERATIONS: DelegationSkill(
        action=TaskAction.JIRA_OPERATIONS,
        name="Jira Operations",
        description="Search Jira tickets/customers, create or transition tickets",
        tags=["jira", "ticket", "delegate"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        handler=make_a2a_delegate_handler(short_name="jira", url_attr="JIRA_AGENT_A2A_URL"),
    ),
    TaskAction.TELEMETRY_QUERY: DelegationSkill(
        action=TaskAction.TELEMETRY_QUERY,
        name="Telemetry Query",
        description="Query network telemetry (InfluxDB) for traffic, interface state, errors",
        tags=["telemetry", "influx", "delegate"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        handler=make_a2a_delegate_handler(short_name="telemetry", url_attr="TELEMETRY_AGENT_A2A_URL"),
    ),
    TaskAction.ALARM_QUERY: DelegationSkill(
        action=TaskAction.ALARM_QUERY,
        name="Alarm Query",
        description="Query Zabbix alarms, events, problems, maintenance windows",
        tags=["alarms", "zabbix", "delegate"],
        memory_scope=MemoryScope.LIGHTWEIGHT,
        handler=make_a2a_delegate_handler(short_name="alarming", url_attr="ALARMING_AGENT_A2A_URL"),
    ),
}
