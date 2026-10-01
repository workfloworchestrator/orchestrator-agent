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

"""Workflow form-fill over A2A: a deterministic walk over core's pages, with a model reading what the caller says."""

from pydantic_ai.models import Model

from orchestrator_agent.form_fill.capability import FormFillCapability
from orchestrator_agent.form_fill.interpret import Interpretation, Interpreter, ModelInterpreter
from orchestrator_agent.form_fill.skill import CallTool, FormFillSkill
from orchestrator_agent.state import FormReply, Reply


def build_form_fill_skill(model: Model | str | None = None) -> FormFillSkill:
    """The skill as configured: the literal contract, with ``model`` interpreting a person's answers core rejected.

    The interpreter is one protocol attribute; the Jev branch puts its decision engine behind the same protocol.
    """
    return FormFillSkill(interpret=ModelInterpreter(model) if model is not None else None)


__all__ = [
    "CallTool",
    "FormFillCapability",
    "FormFillSkill",
    "FormReply",
    "Interpretation",
    "Interpreter",
    "ModelInterpreter",
    "Reply",
    "build_form_fill_skill",
]
