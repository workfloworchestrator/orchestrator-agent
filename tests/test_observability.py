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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from orchestrator_agent import observability


def _install_fake_langfuse(monkeypatch, *, auth_ok: bool) -> MagicMock:
    """Inject a fake `langfuse` module exposing get_client(); return the client mock."""
    client = MagicMock()
    client.auth_check.return_value = auth_ok
    fake_module = SimpleNamespace(get_client=lambda: client)
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return client


def test_configure_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(observability.agent_settings, "LANGFUSE_ENABLED", False)
    instrument = MagicMock()
    monkeypatch.setattr("pydantic_ai.Agent.instrument_all", instrument)

    assert observability.configure_langfuse() is None
    instrument.assert_not_called()


def test_configure_returns_none_when_extra_missing(monkeypatch):
    monkeypatch.setattr(observability.agent_settings, "LANGFUSE_ENABLED", True)
    # Setting the module to None forces `import langfuse` to raise ImportError.
    monkeypatch.setitem(sys.modules, "langfuse", None)

    assert observability.configure_langfuse() is None


def test_configure_returns_none_when_auth_fails(monkeypatch):
    monkeypatch.setattr(observability.agent_settings, "LANGFUSE_ENABLED", True)
    client = _install_fake_langfuse(monkeypatch, auth_ok=False)
    instrument = MagicMock()
    monkeypatch.setattr("pydantic_ai.Agent.instrument_all", instrument)

    assert observability.configure_langfuse() is None
    client.auth_check.assert_called_once()
    instrument.assert_not_called()


def test_configure_instruments_and_returns_client_on_success(monkeypatch):
    monkeypatch.setattr(observability.agent_settings, "LANGFUSE_ENABLED", True)
    client = _install_fake_langfuse(monkeypatch, auth_ok=True)
    instrument = MagicMock()
    monkeypatch.setattr("pydantic_ai.Agent.instrument_all", instrument)

    result = observability.configure_langfuse()

    assert result is client
    instrument.assert_called_once_with()


def test_shutdown_flushes_client():
    client = MagicMock()
    observability.shutdown_langfuse(client)
    client.flush.assert_called_once_with()
