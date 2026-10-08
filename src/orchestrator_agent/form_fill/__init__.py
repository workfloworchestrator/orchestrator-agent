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

"""Workflow form-fill: a deterministic walk over core's pages, answered by a human through native pauses.

The pauses travel over A2A (kagent's human-in-the-loop extension) or chat completions (LibreChat's ask-user tool).
"""

from collections.abc import Sequence

from pydantic_ai.models import Model

from orchestrator_agent.form_fill.capability import FormFillCapability
from orchestrator_agent.form_fill.interpret import Interpreter, ModelInterpreter
from orchestrator_agent.form_fill.labels import CoreLabels, core_api_url
from orchestrator_agent.form_fill.skill import CallTool, FormFillSkill
from orchestrator_agent.form_fill.widgets import (
    BUILTIN_WIDGETS,
    CoreGraphQL,
    FieldWidget,
    GraphQL,
    build_widgets,
    graphql_url,
    load_extender,
)
from orchestrator_agent.settings import agent_settings
from orchestrator_agent.state import Approval, AskField, FormInput, FormReply, Reply


def build_form_fill_skill(
    model: Model | str | None = None,
    *,
    widgets: Sequence[FieldWidget] | None = None,
    graphql: GraphQL | None = None,
) -> FormFillSkill:
    """The skill as configured, with ``model`` reading what a person typed where a value was expected.

    ``model`` interprets typed answers core rejected and chooses the option typed words mean for a long-list
    widget field. The widgets are the built-ins as ``FORM_WIDGET_EXTENDER`` arranges them (a bad extender
    fails here, at startup); they read core's GraphQL API beside its MCP endpoint unless configured otherwise.
    Questions are worded with the labels the frontend shows, core's form translations beside its MCP endpoint.
    """
    reader = ModelInterpreter(model) if model is not None else None
    return FormFillSkill(
        interpret=reader,
        widgets=build_widgets(BUILTIN_WIDGETS, load_extender(agent_settings.FORM_WIDGET_EXTENDER))
        if widgets is None
        else list(widgets),
        graphql=graphql
        or CoreGraphQL(graphql_url(agent_settings.WFO_CORE_MCP_URL, agent_settings.WFO_CORE_GRAPHQL_URL)),
        choose=reader,
        labels=CoreLabels(core_api_url(agent_settings.WFO_CORE_MCP_URL)),
    )


__all__ = [
    "Approval",
    "AskField",
    "CallTool",
    "FormFillCapability",
    "FormFillSkill",
    "FormInput",
    "FormReply",
    "Interpreter",
    "ModelInterpreter",
    "Reply",
    "build_form_fill_skill",
]
