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

"""Regression tests for AgentAdapter.run_stream_events streaming contract."""

from pydantic_ai import AgentEventStream
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.models.test import TestModel

from orchestrator_agent.agent import AgentAdapter
from orchestrator_agent.state import SearchState


def test_run_stream_events_returns_agent_event_stream():
    """pydantic-ai's AG-UI transport drives the stream as an async context manager.

    ``run_stream_events()`` must therefore return an ``AgentEventStream`` (not a bare async
    generator), otherwise ``async with agent.run_stream_events(...)`` raises
    "'async_generator' object does not support the asynchronous context manager protocol".
    """
    agent = AgentAdapter(model=TestModel())
    stream = agent.run_stream_events(deps=StateDeps(SearchState(user_input="hello")))
    assert isinstance(stream, AgentEventStream)


async def test_run_stream_events_supports_async_with():
    """The AG-UI consumption pattern (``async with ... as stream`` then ``async for``) must work."""
    agent = AgentAdapter(model=TestModel())

    events = []
    async with agent.run_stream_events(deps=StateDeps(SearchState(user_input="hello"))) as stream:
        async for event in stream:
            events.append(event)

    assert events, "expected the agent to stream at least one event"
