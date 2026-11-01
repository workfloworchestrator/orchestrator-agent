"""Widgets: what the agent knows about the form fields whose options core leaves to a client."""

from orchestrator_agent.form_fill.widgets.base import (
    MAX_CANDIDATES,
    MAX_FULL_READ,
    MAX_INLINE,
    Fetch,
    Option,
    WidgetContext,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.enrich import EnrichedPage, LongList, enrich, is_widget_field, widget_target
from orchestrator_agent.form_fill.widgets.resolve import (
    Resolution,
    exact_options,
    resolve_answer,
    resolve_words,
    shown_as,
)
from orchestrator_agent.form_fill.widgets.source import (
    CoreApi,
    SourceError,
    api_base,
    can_search,
    source_options,
    sourced,
)

__all__ = [
    "MAX_CANDIDATES",
    "MAX_FULL_READ",
    "MAX_INLINE",
    "CoreApi",
    "EnrichedPage",
    "Fetch",
    "LongList",
    "Option",
    "Resolution",
    "SourceError",
    "WidgetContext",
    "api_base",
    "can_search",
    "enrich",
    "exact_options",
    "is_widget_field",
    "names_of",
    "narrow_options",
    "resolve_answer",
    "resolve_words",
    "shown_as",
    "source_options",
    "sourced",
    "widget_target",
]
