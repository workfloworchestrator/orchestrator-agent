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

from textwrap import dedent

from orchestrator.core.search.core.types import EntityType

from orchestrator_agent import tools
from orchestrator_agent.state import SearchState

AGENT_CONTEXT = """You are an agent that executes tasks in a plan, one step at a time.
When tools complete successfully, results are immediately streamed to the user's UI in real-time."""

FILTERING_RULES = f"""### Filtering Rules (if query requires filters)
- **MANDATORY FIRST STEP**: You MUST call `{tools.discover_filter_paths.__name__}` BEFORE calling `{tools.set_filter_tree.__name__}`. Never skip this — filter paths are database-specific and cannot be guessed.
- Pass simple field names to discovery (e.g. "status", "id", "start_date") — not dotted paths like "subscription.status"
- **USE EXACT PATHS**: Only use paths returned by `{tools.discover_filter_paths.__name__}`. Do not modify or invent paths.
- **MATCH OPERATORS**: Only use operators compatible with the field type as confirmed by `{tools.get_valid_operators.__name__}`
- **PREFER LENIENT OPERATORS** — choose the broadest operator that still captures the user's intent:
  - Text, names, titles, descriptions, or partial values → use `like` (substring match, e.g. `%acme%`), NOT `eq`. Reserve `eq` for exact identifiers (UUIDs), enum/status values, and booleans.
  - Dates and numbers ("in 2025", "after X", "between X and Y", "more than 100") → use range operators `between`/`gt`/`gte`/`lt`/`lte`, NOT `eq`.
  - Avoid `eq` on human-typed text: an over-strict filter that matches nothing is worse than a broad one.
- **KEEP KNOWN STRUCTURED FILTERS**: When the user names a concrete dimension like status or product, always include it as a filter — even when you also match on free text. Filters narrow the candidate set *before* ranking, so they make results more relevant, not fewer. Use `eq` when the exact value is known (e.g. status `active`); use `like` when unsure of the exact stored value (e.g. a product name).
- **EXTRACT IDENTIFIERS**: Scan the request for specific identifiers the user gave — entity/subscription ids, customer names, reference codes (e.g. `IS4443`), or numbers (e.g. `4433`, `id 1234`). These are the highest-signal part of the request. Use `{tools.discover_filter_paths.__name__}` to find the field that holds such a value and filter it with `like` (substring/typo-tolerant). If no discovered field clearly matches an opaque identifier, do NOT invent a filter — the search already ranks on the full request text. Never silently ignore an identifier the user provided.
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

        **Note:** If a search returns no results, the system automatically retries with a broader semantic search (filters dropped) and shows the closest matches. Don't over-constrain your filters. If the result description says matches are approximate, briefly tell the user the results are the closest available rather than exact matches.

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

        ## Example
        Request: "Find X and export them"
        Plan: {{"tasks": [{{"action_type": "search", "reasoning": "Search for X"}}, {{"action_type": "result_actions", "reasoning": "Export the results"}}]}}

        Note: Getting detailed data for a single entity by its id or id-prefix (when the entity type is stated) requires a single RESULT_ACTIONS task, not SEARCH. Preparing an export also requires a RESULT_ACTIONS task."""

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
        - If the user references a specific entity by id or id-prefix: Call {tools.get_entity_by_id.__name__}(id_or_prefix=..., entity_type=...)
        - If a full known UUID is already in hand (e.g. from a previous result): Call {tools.fetch_entity_details.__name__}(entity_id=..., entity_type=...)

        ## Your Task
        Execute the requested action. After calling the tool, respond with a single short confirmation.

        IMPORTANT: For export requests, ONLY call prepare_export(). Do NOT fetch entity details.

        ---

        {context}
        """
    ).strip()
