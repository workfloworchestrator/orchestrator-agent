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

# --- Tool argument keys that reference filterable field paths -------------------------------
# A tool call that supplies any of these (``filters`` on search/aggregate, ``group_by`` on
# aggregate) references database-specific field paths that cannot be guessed — they must come
# from a prior DISCOVER_FILTER_PATHS_TOOL call. FilterPathGuard gates on the
# *presence of these parameters in a tool's own schema*, so it needs no hardcoded tool-name list.
PATH_CONSUMING_PARAMS = ("filters", "group_by")


__all__ = [
    "AGGREGATE_TOOL",
    "DISCOVER_FILTER_PATHS_TOOL",
    "EXPORT_QUERY_TOOL",
    "GET_VALID_OPERATORS_TOOL",
    "PATH_CONSUMING_PARAMS",
    "RESOLVE_ENTITY_TOOL",
    "SEARCH_TOOL",
]
