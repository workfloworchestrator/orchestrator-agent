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

"""The agent as an OpenAI-compatible chat model, and the clients that can show a person a workflow form over it.

``completions`` is the endpoint (``/v1/chat/completions``), ``request`` the request as it is read;
``librechat`` is LibreChat as the client — its ask-user tool, the one way a form is shown over chat
completions today. Another chat client's way of asking a person is a module next to it, added to the
adapter's transports.
"""

from orchestrator_agent.adapters.chat.completions import (
    CLIENT_HEADER,
    CONVERSATION_HEADER,
    ChatCompletionsAdapter,
    ChatHitl,
)
from orchestrator_agent.adapters.chat.librechat import LibreChatHitl
from orchestrator_agent.adapters.chat.request import ChatRequest

__all__ = [
    "CLIENT_HEADER",
    "CONVERSATION_HEADER",
    "ChatCompletionsAdapter",
    "ChatHitl",
    "ChatRequest",
    "LibreChatHitl",
]
