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

"""The agent over A2A: the protocol adapter, and the clients that can show a person a workflow form over it.

``adapter`` is the endpoint (agent card, JSON-RPC, the executor); ``kagent`` is kagent as the client —
its human-in-the-loop extension, the one way a form is shown over A2A today. Another client's way of
pausing a task is a module next to it, added to the executor's transports.
"""

from orchestrator_agent.adapters.a2a.adapter import (
    A2A_SKILLS,
    AGENT_CARD_DESCRIPTION,
    A2AAdapter,
    A2AHitl,
    WFOAgentExecutor,
)
from orchestrator_agent.adapters.a2a.kagent import KagentHitl
from orchestrator_agent.turn import NO_RESULTS

__all__ = [
    "A2A_SKILLS",
    "AGENT_CARD_DESCRIPTION",
    "NO_RESULTS",
    "A2AAdapter",
    "A2AHitl",
    "KagentHitl",
    "WFOAgentExecutor",
]
