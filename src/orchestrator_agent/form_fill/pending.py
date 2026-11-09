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

"""The stop a form waits on — the start to approve — kept so the person's response can be mapped back.

The approval goes out through a human-in-the-loop transport and its answer arrives later, on another
request. What has to be remembered in between is the same whichever transport shows it: the stop's
``id``, which the response names, and how the client was shown it, for a client that reopens the
conversation and must show it again. It is stored on the session (``FormFillSession.pending``); no
transport keeps a model of its own.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from orchestrator_agent.state import FormFillSession


class PendingStop(BaseModel):
    """The approval the form waits on (stored on the session)."""

    id: str
    # The stop as the transport showed it, for a client that reopens the conversation and must show it again.
    shown: dict[str, Any] | None = None


def pending_approval(request_id: str) -> PendingStop:
    """What to remember of the start to approve: its id."""
    return PendingStop(id=request_id)


def pending_of(session: FormFillSession | None) -> PendingStop | None:
    """The stop the session is paused at; None when there is none (or what is stored is not one)."""
    if session is None or not session.pending:
        return None
    try:
        return PendingStop.model_validate(session.pending)
    except ValidationError:
        return None


__all__ = ["PendingStop", "pending_approval", "pending_of"]
