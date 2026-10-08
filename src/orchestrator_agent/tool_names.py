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

"""Single source of truth for orchestrator-core's MCP tool contract.

These names are the ``operation_id``s the core MCP server exposes (see core's
``api/api_v1/endpoints/mcp_tools.py``). Every other module — prompts, capability
specs, artifact mapping, the deterministic-search guard — imports the constants
from here instead of repeating string literals, so the tool contract is declared
in exactly one place.
"""

from __future__ import annotations

# --- MCP tool names (operation_id on core's MCP server) ------------------------------------
SEARCH_TOOL = "search"
AGGREGATE_TOOL = "aggregate"
DISCOVER_FILTER_PATHS_TOOL = "discover_filter_paths"
GET_VALID_OPERATORS_TOOL = "get_valid_operators"
RESOLVE_ENTITY_TOOL = "resolve_entity"
EXPORT_QUERY_TOOL = "export_query"
# Workflow form-fill: walk a workflow's multi-page input form, then start it.
LIST_WORKFLOWS_TOOL = "list_workflows"
GET_WORKFLOW_FORM_TOOL = "get_workflow_form"
CREATE_WORKFLOW_TOOL = "create_workflow"
RESUME_WORKFLOW_PROCESS_TOOL = "resume_workflow_process"
ABORT_WORKFLOW_PROCESS_TOOL = "abort_workflow_process"
SUBSCRIPTION_WORKFLOWS_TOOL = "get_subscription_available_workflows"
# The options of a ``productId`` form field (form-fill widgets).
LIST_PRODUCTS_TOOL = "list_products"
# Local (non-MCP) tool: the model hands a workflow start to the form-fill skill; not in ALL_TOOL_NAMES.
START_WORKFLOW_FORM_TOOL = "start_workflow_form"

# Parameters of the workflow tools above, as their MCP schemas name them.
# ``create_workflow`` is a path-plus-body endpoint in core with no request model to build its arguments from;
# every other tool's arguments come from core's request models (``orchestrator.core.schemas.mcp_tools``).
WORKFLOW_KEY_PARAM = "workflow_key"
JSON_DATA_PARAM = "json_data"

# Every tool name the agent code depends on (artifact mapping, the filter-path guard).
# Verified against the live MCP server at startup
# (``verify_tool_contract``) so a rename in orchestrator-core fails loudly instead of silently
# breaking artifact mapping or leaving the prompts pointing at a tool that no longer exists.
ALL_TOOL_NAMES = (
    SEARCH_TOOL,
    AGGREGATE_TOOL,
    DISCOVER_FILTER_PATHS_TOOL,
    GET_VALID_OPERATORS_TOOL,
    RESOLVE_ENTITY_TOOL,
    EXPORT_QUERY_TOOL,
    LIST_WORKFLOWS_TOOL,
    GET_WORKFLOW_FORM_TOOL,
    CREATE_WORKFLOW_TOOL,
    RESUME_WORKFLOW_PROCESS_TOOL,
    ABORT_WORKFLOW_PROCESS_TOOL,
    SUBSCRIPTION_WORKFLOWS_TOOL,
    LIST_PRODUCTS_TOOL,
)

# Tools that change the orchestrator. Hidden from the model by ``WriteToolGate``; the deterministic form-fill
# skill calls ``create_workflow`` from code after the caller confirmed. Resume/abort are hidden too and not
# offered through the agent yet.
WRITE_TOOL_NAMES = (CREATE_WORKFLOW_TOOL, RESUME_WORKFLOW_PROCESS_TOOL, ABORT_WORKFLOW_PROCESS_TOOL)

# Map of ``CONSTANT_NAME -> live tool name``. A plugin declares the tools it owns in frontmatter by
# *constant* (``tools: [SEARCH_TOOL]``); ``behavior.owned_tool_names`` resolves them to live names
# for tool ownership (the artifact filter + DeferredToolGate). Prompts do not name tools at all — the
# model binds intent to a tool from the tool's description — so this is the single source of truth for
# the constant->name mapping. Auto-derived from the ``*_TOOL`` constants above.
TOOL_NAME_PLACEHOLDERS = {
    name: value for name, value in dict(globals()).items() if name.endswith("_TOOL") and isinstance(value, str)
}

# --- Tool argument keys that reference filterable field paths -------------------------------
# A tool call that supplies any of these (``filters`` on search/aggregate, ``group_by`` on
# aggregate) references database-specific field paths that cannot be guessed — they must come
# from a prior DISCOVER_FILTER_PATHS_TOOL call. FilterPathGuard gates on the
# *presence of these parameters in a tool's own schema*, so it needs no hardcoded tool-name list.
PATH_CONSUMING_PARAMS = ("filters", "group_by")


__all__ = [
    "ALL_TOOL_NAMES",
    "TOOL_NAME_PLACEHOLDERS",
    "ABORT_WORKFLOW_PROCESS_TOOL",
    "JSON_DATA_PARAM",
    "WORKFLOW_KEY_PARAM",
    "AGGREGATE_TOOL",
    "CREATE_WORKFLOW_TOOL",
    "DISCOVER_FILTER_PATHS_TOOL",
    "EXPORT_QUERY_TOOL",
    "GET_VALID_OPERATORS_TOOL",
    "GET_WORKFLOW_FORM_TOOL",
    "LIST_PRODUCTS_TOOL",
    "LIST_WORKFLOWS_TOOL",
    "PATH_CONSUMING_PARAMS",
    "RESOLVE_ENTITY_TOOL",
    "RESUME_WORKFLOW_PROCESS_TOOL",
    "SEARCH_TOOL",
    "START_WORKFLOW_FORM_TOOL",
    "SUBSCRIPTION_WORKFLOWS_TOOL",
    "WRITE_TOOL_NAMES",
]
