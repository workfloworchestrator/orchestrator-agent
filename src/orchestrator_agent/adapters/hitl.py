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

"""What every human-in-the-loop transport does, whichever client it speaks to.

The form-fill skill knows no client: a stop of it is a ``Reply`` (a page's questions, or the start to
approve) and what continues it is a ``FormInput``. A transport is the translation for one client — kagent
through its A2A extension, LibreChat through its ask-user tool — and they all do the same three things:
tell whether a request comes from their client in a state where it can show a person a stop, turn a stop
into what makes the client show it, and read the person's response back.

The wire formats stay each client's own (they are the clients' contracts, and have nothing in common);
what is shared is this shape, so a protocol adapter picks the transport of the client that is calling.
A transport lives with the protocol it speaks (``adapters.a2a.kagent``, ``adapters.chat.librechat``): a new
client is a new module there, added to that adapter's transports.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from orchestrator_agent.state import FormFillSession, FormInput, Reply

# The request of the protocol the client speaks: an A2A ``RequestContext``, a chat-completions call.
RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
# What that protocol's adapter sends back to make the client show a stop: over A2A the metadata of an
# ``input-required`` status message (the stop's payload under the URI of the extension that defines it),
# over chat completions an assistant message.
StopT_co = TypeVar("StopT_co", covariant=True)


class HumanInTheLoop(Protocol[RequestT_contra, StopT_co]):
    """One client's way of showing a person a stop of the form-fill skill and bringing their response back."""

    def shows_stops(self, request: RequestT_contra) -> bool:
        """Whether the request comes from this client, in a state where it can show a person a stop."""
        ...

    def read(self, session: FormFillSession | None, request: RequestT_contra) -> FormInput | StopT_co | None:
        """The person's response carried by the request, as the skill's input; None when it carries none.

        A ``StopT`` instead means the transport answers this request itself and the skill does not run:
        the client is not done with the pending stop (there is more of it to show, or it could not show it).
        """
        ...

    def pause(self, reply: Reply, session: FormFillSession | None, request: RequestT_contra) -> StopT_co | None:
        """The reply's stop as what makes the client show it; None when the reply ends the form.

        What was asked is remembered on the session, so this must run before the state is persisted.
        """
        ...

    def words(self, reply: Reply) -> str:
        """The reply's text for whoever reads it through this client."""
        ...


__all__ = ["HumanInTheLoop"]
