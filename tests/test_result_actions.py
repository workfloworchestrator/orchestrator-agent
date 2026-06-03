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
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturn

import orchestrator_agent.tools.result_actions as ra
from orchestrator_agent.artifacts import DataArtifact
from orchestrator_agent.entity_lookup import ResolvedEntity
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


FULL_UUID = "12345678-1234-5678-1234-567812345678"


class _ResolverRecorder:
    """Stub for resolve_entity_id_prefix: records calls, returns canned matches."""

    def __init__(self, returns):
        self.returns = returns
        self.calls = []

    def __call__(self, session, entity_type, prefix, limit):
        self.calls.append((entity_type, prefix, limit))
        return self.returns


def _stub_fetch(monkeypatch):
    """Replace _fetch_entity_detail with a recording sentinel returning a DataArtifact."""
    calls = []

    async def fake_fetch(ctx, entity_id, entity_type):
        calls.append((entity_id, entity_type))
        return ToolReturn(
            return_value="SENTINEL",
            metadata=DataArtifact(description="d", entity_id=entity_id, entity_type=entity_type.value),
        )

    monkeypatch.setattr(ra, "_fetch_entity_detail", fake_fetch)
    return calls


def _patch_db(monkeypatch):
    """get_entity_by_id reads db.session; the resolver is stubbed so the session is unused."""
    monkeypatch.setattr(ra, "db", SimpleNamespace(session=object()))


class TestGetEntityById:
    async def test_full_uuid_delegates_without_resolving(self, monkeypatch):
        _patch_db(monkeypatch)
        fetch_calls = _stub_fetch(monkeypatch)
        resolver = _ResolverRecorder([])
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", resolver)
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, FULL_UUID, EntityType.SUBSCRIPTION)

        assert result.return_value == "SENTINEL"
        assert fetch_calls == [(FULL_UUID, EntityType.SUBSCRIPTION)]
        assert resolver.calls == []

    async def test_single_prefix_match_delegates_to_fetch(self, monkeypatch):
        _patch_db(monkeypatch)
        fetch_calls = _stub_fetch(monkeypatch)
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", _ResolverRecorder([ResolvedEntity(FULL_UUID, "Acme")]))
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, "abcd1234", EntityType.SUBSCRIPTION)

        assert result.return_value == "SENTINEL"
        assert fetch_calls == [(FULL_UUID, EntityType.SUBSCRIPTION)]

    async def test_multiple_matches_return_plain_candidate_list(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        matches = [ResolvedEntity("aaaa1111", "Acme"), ResolvedEntity("aaaa2222", "Beta")]
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", _ResolverRecorder(matches))
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, "aaaa", EntityType.SUBSCRIPTION)

        assert result.metadata is None
        assert result.return_value["candidates"] == [
            {"entity_id": "aaaa1111", "title": "Acme"},
            {"entity_id": "aaaa2222", "title": "Beta"},
        ]
        assert "refine" in result.return_value["instruction"].lower()

    async def test_more_than_limit_matches_caps_and_notes(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        matches = [ResolvedEntity(f"aaaa{i:04d}", f"Title {i}") for i in range(11)]
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", _ResolverRecorder(matches))
        ctx = _ctx_with_active_step()

        result = await ra.get_entity_by_id(ctx, "aaaa", EntityType.SUBSCRIPTION)

        assert len(result.return_value["candidates"]) == 10
        assert "first 10" in result.return_value["instruction"]

    async def test_zero_matches_raises_model_retry(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", _ResolverRecorder([]))
        ctx = _ctx_with_active_step()

        with pytest.raises(ModelRetry, match="No SUBSCRIPTION found with id starting with abcd"):
            await ra.get_entity_by_id(ctx, "abcd1234", EntityType.SUBSCRIPTION)

    async def test_too_short_prefix_raises_without_resolving(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        resolver = _ResolverRecorder([])
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", resolver)
        ctx = _ctx_with_active_step()

        with pytest.raises(ModelRetry, match="at least 4 characters"):
            await ra.get_entity_by_id(ctx, "abc", EntityType.SUBSCRIPTION)
        assert resolver.calls == []

    async def test_non_hex_raises_search_hint(self, monkeypatch):
        _patch_db(monkeypatch)
        _stub_fetch(monkeypatch)
        monkeypatch.setattr(ra, "resolve_entity_id_prefix", _ResolverRecorder([]))
        ctx = _ctx_with_active_step()

        with pytest.raises(ModelRetry, match="search by name"):
            await ra.get_entity_by_id(ctx, "acme corp", EntityType.SUBSCRIPTION)


def test_get_entity_by_id_is_exported_from_tools_package():
    from orchestrator_agent import tools

    assert hasattr(tools, "get_entity_by_id")
    assert "get_entity_by_id" in tools.__all__
