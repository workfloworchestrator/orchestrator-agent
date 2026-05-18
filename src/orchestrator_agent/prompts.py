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

import json
from textwrap import dedent
from typing import Any

from orchestrator.search.core.types import EntityType

from orchestrator_agent import tools
from orchestrator_agent.state import SearchState

AGENT_CONTEXT = """You are an agent that executes tasks in a plan, one step at a time.
When tools complete successfully, results are immediately streamed to the user's UI in real-time."""

FILTERING_RULES = f"""### Filtering Rules (if query requires filters)
- **MANDATORY FIRST STEP**: You MUST call `{tools.discover_filter_paths.__name__}` BEFORE calling `{tools.set_filter_tree.__name__}`. Never skip this — filter paths are database-specific and cannot be guessed.
- Pass simple field names to discovery (e.g. "status", "id", "start_date") — not dotted paths like "subscription.status"
- **USE EXACT PATHS**: Only use paths returned by `{tools.discover_filter_paths.__name__}`. Do not modify or invent paths.
- **MATCH OPERATORS**: Only use operators compatible with the field type as confirmed by `{tools.get_valid_operators.__name__}`
- Temporal constraints like "in 2025", "between X and Y" require filters on datetime fields
- If a discovered path does not match the user's intent, try alternative field names in a new discovery call"""


def get_search_execution_prompt(state: SearchState) -> str:
    """Get prompt for Search skill.

    Args:
        state: Current search state

    Returns:
        Complete prompt for executing search with optional filtering
    """
    context = state.memory.format_context_for_llm(state)

    return dedent(
        f"""
        # Searching

        {AGENT_CONTEXT}

        ## Your Task
        Execute a database search to answer the user's request.
        **IMPORTANT**: This query starts empty - previous query filters shown in history are NOT applied unless you rebuild them.

        ## Steps
        1. Determine the entity_type for this search (SUBSCRIPTION, PRODUCT, WORKFLOW, or PROCESS)
        2. If filters needed (almost always):
           a. Call `{tools.discover_filter_paths.__name__}(field_names=[...], entity_type=...)` to get valid paths
           b. Call `{tools.get_valid_operators.__name__}` to confirm valid operators for the field type
           c. Build FilterTree using ONLY the exact paths from step 2a
           d. Call `{tools.set_filter_tree.__name__}` with the validated FilterTree
        3. Call {tools.run_search.__name__}(entity_type=...) — you MUST pass entity_type
        4. Explain what you did in 1-2 sentences at most. DO NOT list the actual results, they are already shown to the user.

        {FILTERING_RULES}

        ---

        {context}
    """
    ).strip()


def get_aggregation_execution_prompt(state: SearchState) -> str:
    """Get prompt for Aggregation skill.

    Args:
        state: Current search state with query_operation and query info

    Returns:
        Complete prompt for executing aggregation with optional filtering and grouping
    """
    context = state.memory.format_context_for_llm(state)

    return dedent(
        f"""
        # Aggregating

        {AGENT_CONTEXT}

        ## Your Task
        Execute an aggregation query for the user's request.
        **IMPORTANT**: This query starts empty - previous query filters/grouping shown in history are NOT applied unless you rebuild them.

        ## Steps
        1. Determine entity_type (SUBSCRIPTION, PRODUCT, WORKFLOW, or PROCESS) and query_operation (COUNT for counting, AGGREGATE for numeric calculations like SUM/AVG/MIN/MAX)
        2. If filters needed:
           a. Call `{tools.discover_filter_paths.__name__}(field_names=[...], entity_type=...)` to get valid paths
           b. Call `{tools.get_valid_operators.__name__}` to confirm valid operators for the field type
           c. Build FilterTree using ONLY the exact paths from step 2a
           d. Call `{tools.set_filter_tree.__name__}` with the validated FilterTree
        3. Set grouping: Temporal ({tools.set_temporal_grouping.__name__}) or regular ({tools.set_grouping.__name__}) — you MUST pass entity_type and query_operation
        4. For AGGREGATE operation ONLY: Call {tools.set_aggregations.__name__}(entity_type=..., query_operation=...). For COUNT: Do NOT call (counting is automatic)
        5. Call {tools.run_aggregation.__name__}(entity_type=..., query_operation=..., visualization_type=...)
        6. Explain what you did in 1-2 sentences at most. DO NOT list the actual results, they are already shown to the user

        {FILTERING_RULES}
        - Filters restrict WHICH records; grouping controls HOW to aggregate

        ---

        {context}
    """
    ).strip()


def get_text_response_prompt(state: SearchState) -> str:
    """Get prompt for TextResponseNode agent.

    Args:
        state: Current search state

    Returns:
        Complete prompt for generating text response
    """
    context = state.memory.format_context_for_llm(state)
    entity_types = ", ".join([et.value for et in EntityType])

    return dedent(
        f"""
        # Responding

        {AGENT_CONTEXT}

        ## Available Capabilities
        - Search for entities: {entity_types}
        - Filter searches by various criteria (status, dates, custom fields)
        - Count and aggregate data (totals, averages, grouping by fields or time periods)
        - Return structured data with visualization hints (table, bar chart, line chart, etc.)
        - Export search results
        - Fetch detailed information about specific entities

        ## Your Task
        Generate a helpful response to the user's question.

        ---

        {context}
    """
    ).strip()


def get_workflow_form_fill_prompt(state: SearchState) -> str:
    """Prompt for the WORKFLOW_FORM_FILL skill.

    Routes between four tools based on whether a session is active and what
    the user/adapter has provided this turn.
    """
    context = state.memory.format_context_for_llm(state)
    session = state.form_fill
    submission = (state.adapter_metadata or {}).get("form_submission")

    session_json = session.model_dump_json(indent=2) if session else "null"
    submission_json = json.dumps(submission, indent=2) if submission else "null"

    return dedent(
        f"""
        # Workflow Form Fill

        {AGENT_CONTEXT}

        ## Your Task
        Drive an orchestrator-core workflow form to completion.

        ## Tools
        - `{tools.start_workflow_form.__name__}(workflow_key)` — begin a new form. Use when no session is active.
        - `{tools.submit_workflow_page.__name__}(values=None)` — submit the current page and advance. Reads the submission's `values` when omitted.
        - `{tools.confirm_and_create_workflow.__name__}()` — start the workflow once the user has confirmed the summary.
        - `{tools.cancel_workflow_form.__name__}()` — abort and clear the session.

        ## How to choose
        1. If no active session AND the user just expressed intent: call `{tools.start_workflow_form.__name__}` with the inferred `workflow_key` (snake_case, e.g. `create_lightpath`).
        2. If there is an active session and the form submission's `action == "submit"`: call `{tools.submit_workflow_page.__name__}` (no args; it reads the submission's values).
        3. If the session is in `summary` and the submission's `action == "confirm"` (or user typed yes): call `{tools.confirm_and_create_workflow.__name__}`.
        4. If the submission says cancel: call `{tools.cancel_workflow_form.__name__}`.

        ## Response style after the tool call
        The artifact (form / confirmation prompt) is *already* shown to the user — do NOT
        describe what they can already see. Reply rules:

        - When you called `{tools.start_workflow_form.__name__}` or `{tools.submit_workflow_page.__name__}`
          and the tool emitted a form / confirmation artifact: reply with **exactly the empty
          string** (no text at all). The artifact is the message.
        - When you called `{tools.confirm_and_create_workflow.__name__}`: reply with the
          tool's return_value **verbatim** — it contains the `process_id` the user needs.
          Do not rephrase, summarize, or omit the process_id.
        - When you called `{tools.cancel_workflow_form.__name__}`: reply with the tool's
          return_value verbatim.

        ## Current session
        ```
        form_fill = {session_json}
        form_submission = {submission_json}
        user_input = {state.user_input!r}
        ```

        ---

        {context}
        """
    ).strip()


def get_planning_prompt(state: SearchState) -> str:
    """Get prompt for Planner to create execution plan.

    Args:
        state: Current search state

    Returns:
        Complete prompt for creating multi-step execution plan
    """
    context = state.memory.format_context_for_llm(state)

    guidelines = """## Your Task & Guidelines
        Analyze the user's request and create a sequential execution plan.

        1. **Check available context**: If results already exist from previous turns, you can act on them directly
        2. **Break into tasks**: Each task = one skill execution. Create as many tasks as needed to fulfill the request.

        ## Skill catalog

        Built-in (WFO core, this agent):
        - SEARCH — find subscriptions / products / workflows / processes by filters
        - AGGREGATION — count / sum / avg with grouping
        - RESULT_ACTIONS — export a query or fetch detailed data for a single entity by ID
        - TEXT_RESPONSE — general questions about the system or this agent's capabilities
        - WORKFLOW_FORM_FILL — drive a multi-page workflow form to completion (create / modify / terminate / validate / system tasks)

        Delegated (other domain agents over A2A):
        - IMS_LOOKUP — IMS network inventory: nodes, services, planned-works
        - INCIDENT_LOOKUP — CIM customer incidents: read and create
        - JIRA_OPERATIONS — Jira tickets, customers, locations: read, create, transition, comment, assign
        - TELEMETRY_QUERY — InfluxDB telemetry: traffic counters, interface state, errors, optical metrics
        - ALARM_QUERY — Zabbix alarms: events, problems, maintenance windows

        ## Examples
        Request: "Find X and export them"
        Plan: {{"tasks": [{{"action_type": "search", "reasoning": "Search for X"}}, {{"action_type": "result_actions", "reasoning": "Export the results"}}]}}

        Request: "ik wil een lichtpad" / "create a lightpath" / "I want to start workflow X"
        Plan: {{"tasks": [{{"action_type": "workflow_form_fill", "reasoning": "User wants to start a workflow; drive the form to completion"}}]}}

        Request: "what's the input/output bps of subscription X over the last hour"
        Plan: {{"tasks": [{{"action_type": "telemetry_query", "reasoning": "Pure telemetry question; delegate to telemetry agent"}}]}}

        Request: "are there any open alarms about node Y"
        Plan: {{"tasks": [{{"action_type": "alarm_query", "reasoning": "Pure alarm question; delegate to alarming agent"}}]}}

        Request: "what's the IMS planned maintenance on circuit C-123"
        Plan: {{"tasks": [{{"action_type": "ims_lookup", "reasoning": "IMS-only lookup"}}]}}

        Request: "what's wrong with subscription X" / "diagnose X"
        Plan: {{"tasks": [
          {{"action_type": "search", "reasoning": "Get the subscription details"}},
          {{"action_type": "telemetry_query", "reasoning": "Check current traffic / interface health"}},
          {{"action_type": "alarm_query", "reasoning": "Correlate with any open alarms"}},
          {{"action_type": "jira_operations", "reasoning": "Look for an open ticket about it"}}
        ]}}

        Notes:
        - Getting detailed data for a single entity (by ID) or preparing an export require a RESULT_ACTIONS task, not SEARCH.
        - Any "create / start / I want a [workflow]" intent (English or Dutch) routes to WORKFLOW_FORM_FILL.
        - Pick the most specific delegated skill; don't route a telemetry question through JIRA_OPERATIONS or vice versa.
        - For diagnostic / cross-domain investigations, plan multiple delegated tasks in dependency order.
        - For DELEGATED skills, `reasoning` is forwarded verbatim to the domain agent as its question. Phrase it as a focused, self-contained query that includes every ID / timestamp / filter the domain agent needs."""

    return dedent(
        f"""
        # Execution Planning

        {AGENT_CONTEXT}

        {guidelines}

        IMPORTANT: Query execution skills automatically stream results to the user.
        Do NOT create redundant tasks just to "show" or "present" results that are already displayed.

        ---

        {context}
        """
    ).strip()


def get_synthesis_prompt(state: SearchState, tasks: list[Any], task_results: list[tuple[Any, str]]) -> str:
    """Prompt for the synthesis pass: same agent, fed the executed plan's results.

    Only runs when the plan had more than one task. The synthesizer's job is
    to weave the per-task results into a coherent user-facing answer. Any
    artifacts emitted during task execution have already streamed to the
    surface — the synthesizer is text-only and should refer to them rather
    than repeat their contents.
    """
    pieces: list[str] = []
    for task, output in task_results:
        action = task.action_type.value if hasattr(task.action_type, "value") else str(task.action_type)
        pieces.append(f"- **{action}** ({task.reasoning})\n  result: {output}")
    body = "\n".join(pieces) if pieces else "(no task results captured)"

    return dedent(
        f"""
        # Synthesis

        You previously made a plan and the tasks have now executed.

        ## User's original question
        {state.user_input!r}

        ## Plan results
        {body}

        ## Your task
        Compose ONE coherent answer to the user's question from the results above.
        Be concise. If artifacts have already been emitted by the executed tasks
        (forms, tables, charts), reference them rather than repeat their contents
        (e.g. "the table above shows…", "fill in the form below…"). Do not invent
        details not present in the results. If the results are insufficient to
        answer, say so plainly.
        """
    ).strip()


def get_result_actions_prompt(state: SearchState) -> str:
    """Get prompt for ResultActionsNode agent.

    Args:
        state: Current search state with environment and user input

    Returns:
        Complete prompt for result actions
    """
    context = state.memory.format_context_for_llm(state)
    return dedent(
        f"""
        # Acting on Results

        {AGENT_CONTEXT}

        Act on existing search/aggregation results.

        ## Available Actions
        - If user wants to EXPORT/DOWNLOAD results: Call {tools.prepare_export.__name__}() ONLY
        - If user wants DETAILED INFORMATION about a specific entity: Call {tools.fetch_entity_details.__name__}(entity_id=..., entity_type=...)

        ## Your Task
        Execute the requested action. After calling the tool, respond with a single short confirmation.

        IMPORTANT: For export requests, ONLY call prepare_export(). Do NOT fetch entity details.

        ---

        {context}
        """
    ).strip()
