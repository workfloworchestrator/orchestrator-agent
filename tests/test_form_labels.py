"""Questions worded with the labels the WFO frontend shows (core's form translations), in the order of the page."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from typing import Any

import httpx
import pytest

from orchestrator_agent.form_fill.core_bridge import page_model
from orchestrator_agent.form_fill.labels import NO_LABELS, CoreLabels, FieldLabels, core_api_url
from orchestrator_agent.form_fill.skill import FormFillSkill, questions
from orchestrator_agent.state import SearchState

from .test_form_fill import FakeCore, open_form

PAGE = {
    "properties": {
        "customer_id": {"title": "Customer Id", "type": "string"},
        "contact_persons": {"default": [], "items": {"type": "string"}, "title": "Contact Persons", "type": "array"},
        "ticket_id": {"default": "", "title": "Ticket Id", "type": "string"},
        "port_mode": {"enum": ["tagged", "untagged"], "title": "PortEnum", "type": "string"},
        "lldp": {"default": False, "title": "Lldp", "type": "boolean"},
    },
    "required": ["customer_id", "port_mode"],
    "title": "Service Port 10G",
    "type": "object",
}
TRANSLATIONS = {
    "forms": {
        "fields": {
            "customer_id": "Customer",
            "contact_persons": "Customer contact persons",
            "ticket_id": "Jira ticket ID",
            "port_mode": "Port Mode",
            "port_mode_info": "The port mode of the new service port",
            "lldp": "Enable LLDP",
        }
    }
}
LABELS = FieldLabels(TRANSLATIONS["forms"]["fields"])


def test_questions_follow_the_page_with_the_frontends_labels():
    asked = questions(page_model(PAGE), {}, [], names=LABELS)
    assert [field.name for field in asked] == ["customer_id", "contact_persons", "ticket_id", "port_mode", "lldp"]
    assert [field.title for field in asked] == [
        "Customer",
        "Customer contact persons",
        "Jira ticket ID",
        "Port Mode",
        "Enable LLDP",
    ]
    port_mode = asked[3]
    assert port_mode.question == "Port Mode *" and port_mode.hint == "The port mode of the new service port"
    assert asked[2].question == "Jira ticket ID"  # optional: no mark, no field name


def test_a_field_core_rejected_keeps_its_place_and_says_why():
    errors = [{"loc": ("port_mode",), "msg": "Input should be 'tagged' or 'untagged'", "type": "enum"}]
    asked = questions(page_model(PAGE), {"customer_id": "c-1"}, errors, names=LABELS)
    assert [field.name for field in asked] == ["contact_persons", "ticket_id", "port_mode", "lldp"]
    assert asked[2].question == "Port Mode * — Input should be 'tagged' or 'untagged'"


def test_without_translations_the_schema_title_is_used():
    asked = questions(page_model(PAGE), {}, [], names=NO_LABELS)
    assert asked[0].title == "Customer Id" and asked[0].question == "Customer Id *"


@pytest.mark.parametrize(
    "response,expected",
    [
        pytest.param(httpx.Response(200, json=TRANSLATIONS), "Customer", id="served"),
        pytest.param(httpx.Response(500, text="boom"), None, id="failed"),
        pytest.param(httpx.Response(200, json={"workflow": {}}), None, id="no-form-labels"),
    ],
)
async def test_core_labels_are_fetched_once_and_a_failure_means_no_labels(response, expected):
    calls: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return response

    core = CoreLabels("http://core.test/api", transport=httpx.MockTransport(answer))
    first, second = await core.labels(), await core.labels()
    assert first.title("customer_id") == expected and second is first
    assert calls == ["/api/translations/en-GB"]


async def test_the_skill_words_its_stops_with_core_labels():
    class LabelledCore(FakeCore):
        async def __call__(self, name: str, args: dict[str, Any]) -> Any:
            if name == "get_workflow_form" and not args["page_inputs"]:
                return {"page": 0, "complete": False, "schema": PAGE}
            return await super().__call__(name, args)

    class Labels:
        async def labels(self) -> FieldLabels:
            return LABELS

    skill = FormFillSkill(labels=Labels())
    reply = await open_form(skill, LabelledCore(), SearchState(), "create_demo_lightpath")
    assert reply.question("customer_id").title == "Customer"


@pytest.mark.parametrize(
    "mcp_url,expected",
    [
        pytest.param("http://core:8080/mcp", "http://core:8080/api", id="mcp"),
        pytest.param("http://core:8080/mcp/", "http://core:8080/api", id="trailing-slash"),
    ],
)
def test_the_rest_api_is_beside_the_mcp_endpoint(mcp_url, expected):
    assert core_api_url(mcp_url) == expected
