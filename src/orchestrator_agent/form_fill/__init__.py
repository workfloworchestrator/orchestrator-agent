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

"""Workflow form-fill: the model fills core's form page by page with the person; code keeps the machine.

The pages, the options and the start are code's (``skill``, ``pages``, ``session``); the model fills the
current page through the page tools (``tools``) under the form's instructions (``instructions``). The one
structured stop — the start to approve — travels over A2A (kagent's human-in-the-loop extension), chat
completions (LibreChat's ask-user tool) or AG-UI (an interrupt).
"""

from orchestrator_agent.form_fill.capability import FormFillCapability
from orchestrator_agent.form_fill.pages import CallTool
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.state import Approval, FormInput, FormReply, Reply


def build_form_fill_skill() -> FormFillSkill:
    """The skill as configured."""
    return FormFillSkill()


__all__ = [
    "Approval",
    "CallTool",
    "FormFillCapability",
    "FormFillSkill",
    "FormInput",
    "FormReply",
    "Reply",
    "build_form_fill_skill",
]
