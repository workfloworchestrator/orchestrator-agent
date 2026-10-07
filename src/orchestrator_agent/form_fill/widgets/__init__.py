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
    WIDGET_MARK,
    FieldWidget,
    FieldWidgetExtender,
    GraphQL,
    Option,
    Widget,
    WidgetContext,
    field_hint,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.core import BUILTIN_WIDGETS, CustomerIdWidget, ProductIdWidget
from orchestrator_agent.form_fill.widgets.enrich import EnrichedPage, LongList, enrich
from orchestrator_agent.form_fill.widgets.graphql import CoreGraphQL, GraphQLError, core_auth, graphql_url
from orchestrator_agent.form_fill.widgets.registry import build_widgets, load_extender, match_widget, widget_target
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
    "WIDGET_MARK",
    "CoreGraphQL",
    "CustomerIdWidget",
    "EnrichedPage",
    "FieldWidget",
    "FieldWidgetExtender",
    "GraphQL",
    "GraphQLError",
    "LongList",
    "Option",
    "ProductIdWidget",
    "Resolution",
    "Widget",
    "WidgetContext",
    "build_widgets",
    "core_auth",
    "enrich",
    "exact_options",
    "field_hint",
    "graphql_url",
    "load_extender",
    "match_widget",
    "names_of",
    "narrow_options",
    "resolve_answer",
    "resolve_words",
    "shown_as",
    "widget_target",
]
