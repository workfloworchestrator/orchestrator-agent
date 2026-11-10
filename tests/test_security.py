import pytest
from fastapi import HTTPException
from starlette.requests import Request

from orchestrator_agent.security import subject_of


def _request(**scope) -> Request:
    return Request({"type": "http", **scope})


@pytest.mark.parametrize(
    ("oauth2_active", "scope", "subject"),
    [
        pytest.param(False, {}, None, id="oauth2-off"),
        pytest.param(True, {"state": {"user": {"sub": "alice"}}}, "alice", id="signed-in"),
    ],
)
def test_subject_of_is_the_authenticated_subject(monkeypatch, oauth2_active, scope, subject):
    monkeypatch.setattr("orchestrator_agent.security.oauth2lib_settings.OAUTH2_ACTIVE", oauth2_active)
    assert subject_of(_request(**scope)) == subject


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param({}, id="no-user"),
        pytest.param({"state": {"user": {}}}, id="no-subject"),
        pytest.param({"state": {"user": {"sub": ""}}}, id="empty-subject"),
    ],
)
def test_oauth2_on_refuses_a_token_without_a_subject(monkeypatch, scope):
    monkeypatch.setattr("orchestrator_agent.security.oauth2lib_settings.OAUTH2_ACTIVE", True)
    with pytest.raises(HTTPException) as refused:
        subject_of(_request(**scope))
    assert refused.value.status_code == 401
