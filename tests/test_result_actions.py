# Copyright 2019-2025 SURF, GÉANT.
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

"""Tests for result_actions tools (get_entity_by_id decision logic + fetch refactor)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import httpx
import pytest
from orchestrator.core.search.core.types import EntityType

import orchestrator_agent.tools.result_actions as ra
from orchestrator_agent.artifacts import DataArtifact
from orchestrator_agent.state import SearchState


def _ctx_with_active_step():
    """A RunContext-like object with memory ready to record a ToolStep."""
    state = SearchState(user_input="details please")
    state.memory.start_turn("details please")
    state.memory.start_step("ResultActions")
    return SimpleNamespace(deps=SimpleNamespace(state=state))


async def test_fetch_entity_details_returns_data_artifact(monkeypatch):
    ctx = _ctx_with_active_step()

    response = httpx.Response(
        200,
        json={"id": "e1", "status": "active"},
        request=httpx.Request("GET", "http://orchestrator/api/subscriptions/domain-model/e1"),
    )
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(
        ra, "token_manager", SimpleNamespace(get_token=AsyncMock(return_value=None), auth_enabled=False)
    )
    with patch("orchestrator_agent.tools.result_actions.httpx.AsyncClient", return_value=mock_client):
        result = await ra.fetch_entity_details(ctx, "e1", EntityType.SUBSCRIPTION)

    assert isinstance(result.metadata, DataArtifact)
    assert result.metadata.entity_id == "e1"
    assert result.metadata.entity_type == "SUBSCRIPTION"
    assert json.loads(result.return_value)["status"] == "active"


async def test_fetch_entity_details_404_raises_model_retry(monkeypatch):
    from pydantic_ai.exceptions import ModelRetry

    ctx = _ctx_with_active_step()

    response = httpx.Response(404, request=httpx.Request("GET", "http://orchestrator/api/subscriptions/domain-model/x"))
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(
        ra, "token_manager", SimpleNamespace(get_token=AsyncMock(return_value=None), auth_enabled=False)
    )
    with patch("orchestrator_agent.tools.result_actions.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ModelRetry):
            await ra.fetch_entity_details(ctx, "x", EntityType.SUBSCRIPTION)
