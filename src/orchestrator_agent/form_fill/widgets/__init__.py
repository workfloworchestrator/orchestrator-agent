"""Widgets: what the agent knows about the form fields whose options core leaves to a client."""

from orchestrator_agent.form_fill.widgets.base import (
    MAX_CANDIDATES,
    MAX_FULL_READ,
    MAX_INLINE,
    Fetch,
    FieldWidget,
    Option,
    Widget,
    WidgetContext,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.enrich import EnrichedPage, LongList, enrich, match_widget, widget_target
from orchestrator_agent.form_fill.widgets.resolve import (
    Resolution,
    exact_options,
    resolve_answer,
    resolve_words,
    shown_as,
)
from orchestrator_agent.form_fill.widgets.source import CoreApi, OptionsSourceWidget, SourceError, api_base

__all__ = [
    "MAX_CANDIDATES",
    "MAX_FULL_READ",
    "MAX_INLINE",
    "CoreApi",
    "EnrichedPage",
    "Fetch",
    "FieldWidget",
    "LongList",
    "Option",
    "OptionsSourceWidget",
    "Resolution",
    "SourceError",
    "Widget",
    "WidgetContext",
    "api_base",
    "enrich",
    "exact_options",
    "match_widget",
    "names_of",
    "narrow_options",
    "resolve_answer",
    "resolve_words",
    "shown_as",
    "widget_target",
]
