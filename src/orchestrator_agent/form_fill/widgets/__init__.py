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

"""Widgets: what the agent knows about the form fields whose options core leaves to the frontend."""

from orchestrator_agent.form_fill.widgets.base import (
    MAX_CANDIDATES,
    MAX_FULL_READ,
    MAX_INLINE,
    FieldWidget,
    GraphQL,
    Option,
    Widget,
    WidgetContext,
    field_hint,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.cascade import CascadeWidget, Step
from orchestrator_agent.form_fill.widgets.core import BUILTIN_WIDGETS, CustomerIdWidget, ProductIdWidget
from orchestrator_agent.form_fill.widgets.enrich import EnrichedPage, LongList, enrich, match_widget, widget_target
from orchestrator_agent.form_fill.widgets.graphql import CoreGraphQL, GraphQLError, graphql_url
from orchestrator_agent.form_fill.widgets.resolve import (
    Resolution,
    exact_options,
    resolve_answer,
    resolve_words,
    shown_as,
)

__all__ = [
    "BUILTIN_WIDGETS",
    "MAX_CANDIDATES",
    "MAX_FULL_READ",
    "MAX_INLINE",
    "CascadeWidget",
    "CoreGraphQL",
    "CustomerIdWidget",
    "EnrichedPage",
    "FieldWidget",
    "GraphQL",
    "GraphQLError",
    "LongList",
    "Option",
    "ProductIdWidget",
    "Resolution",
    "Step",
    "Widget",
    "WidgetContext",
    "enrich",
    "exact_options",
    "field_hint",
    "graphql_url",
    "match_widget",
    "names_of",
    "narrow_options",
    "resolve_answer",
    "resolve_words",
    "shown_as",
    "widget_target",
]
