"""A widget field asked in steps — a node first, then a free port on it — the way the frontend's port select works."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.form_fill.widgets import CascadeWidget, Option, Step, WidgetContext
from orchestrator_agent.state import FormFillSession, SearchState

from .test_form_fill import FakeCore, open_form, turn

KEY = "create_port"
PAGE = {
    "properties": {
        "port_id": {
            "format": "imsPortId",
            "title": "Port Id",
            "type": "integer",
            "uniforms": {"interfaceSpeed": 10000, "imsPortMode": "patched"},
        },
        "port_mode": {"enum": ["tagged", "untagged"], "title": "Port Mode", "type": "string"},
    },
    "required": ["port_id", "port_mode"],
    "title": "Service Port 10G",
    "type": "object",
}
NODES = [Option("n-asd", "asd001a-jnx-01", aliases=("asd001a",)), Option("n-ut", "ut002a-jnx-01", aliases=("ut002a",))]
PORTS = {
    "n-asd": [Option(101, "xe-0/0/1 (free) (10GBASE-LR)"), Option(102, "xe-0/0/2 (free) (10GBASE-LR)")],
    "n-ut": [Option(201, "xe-1/0/0 (free) (10GBASE-LR)"), Option(202, "xe-1/0/1 (free) (10GBASE-LR)")],
}


class NodePorts(CascadeWidget):
    """``imsPortId``: a node first, then the free ports of the field's speed on it."""

    id = "imsPortId"

    def __init__(self) -> None:
        self.asked: list[tuple[str, Any]] = []

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("format") == "imsPortId"

    def steps(self, field: Mapping[str, Any]) -> Sequence[Step]:
        return [Step("node", "Node", self.nodes)]

    async def nodes(self, field: Mapping[str, Any], ctx: WidgetContext, chosen: Mapping[str, Any]) -> Sequence[Option]:
        self.asked.append(("node", dict(chosen)))
        return NODES

    async def fetch_chosen(
        self, field: Mapping[str, Any], ctx: WidgetContext, chosen: Mapping[str, Any]
    ) -> Sequence[Option]:
        assert field["uniforms"]["interfaceSpeed"] == 10000
        self.asked.append(("port", dict(chosen)))
        return PORTS[chosen["node"]]


class PortCore:
    """Core's form tools for one page asking a port and its mode; like core, it takes any integer for the port."""

    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        if name == "list_workflows":
            return [{**FakeCore.WORKFLOWS[0], "name": KEY}]
        if name == "create_workflow":
            return {"id": FakeCore.PROCESS_ID}
        inputs = args["page_inputs"]
        self.submitted.extend(inputs)
        if not inputs:
            return {"page": 0, "complete": False, "schema": PAGE}
        FakeCore.require(PAGE, inputs[0])
        return {"page": 1, "complete": True, "schema": None}


class NoFit:
    """A chooser for whom the words fit no option, recording what it was asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def choose(self, title, options, words):
        self.calls.append(words)
        return []


def make(chooser: NoFit | None = None) -> FormFillSkill:
    return FormFillSkill(widgets=[NodePorts()], choose=chooser or NoFit())


async def test_the_node_is_asked_first_with_its_options():
    reply = await open_form(make(), PortCore(), SearchState(), KEY)
    node = reply.question("port_id")
    assert node.title == "Port Id — Node" and node.choices == ("asd001a-jnx-01", "ut002a-jnx-01")
    assert node.values == ("n-asd", "n-ut")


@pytest.mark.parametrize(
    "node_answer",
    [pytest.param("n-asd", id="picked-chip"), pytest.param("asd001a", id="typed-name")],
)
async def test_then_the_free_ports_of_that_node_and_only_the_port_reaches_core(node_answer):
    core, state, chooser = PortCore(), SearchState(), NoFit()
    skill = make(chooser)
    await open_form(skill, core, state, KEY)
    ports = await turn(skill, core, state, {"port_id": node_answer})
    port = ports.question("port_id")
    assert port.title == "Port Id" and port.values == (101, 102)
    assert port.choices == ("xe-0/0/1 (free) (10GBASE-LR)", "xe-0/0/2 (free) (10GBASE-LR)")
    done = await turn(skill, core, state, {"port_id": 102, "port_mode": "tagged"})
    assert done.status == "confirming" and done.values == {"port_id": 102, "port_mode": "tagged"}
    assert done.labels["port_id"] == "xe-0/0/2 (free) (10GBASE-LR)"
    assert all("node" not in page and page.get("port_id") in (None, 102) for page in core.submitted)
    assert chooser.calls == []  # a picked chip and an exact name need no model


async def test_words_that_are_no_node_are_asked_again_and_never_submitted():
    core, state, skill = PortCore(), SearchState(), make()
    await open_form(skill, core, state, KEY)
    again = await turn(skill, core, state, {"port_id": "delft"})
    node = again.question("port_id")
    assert node.title == "Port Id — Node" and "delft" in node.hint
    assert "delft" not in str(core.submitted)


async def test_a_port_that_is_no_option_of_the_node_is_asked_again():
    core, state, skill = PortCore(), SearchState(), make()
    await open_form(skill, core, state, KEY)
    await turn(skill, core, state, {"port_id": "n-ut"})
    again = await turn(skill, core, state, {"port_id": 101, "port_mode": "tagged"})  # a port of the other node
    assert again.status == "gathering" and again.question("port_id").values == (201, 202)
    assert all(page.get("port_id") != 101 for page in core.submitted)


def test_the_chosen_steps_are_kept_with_the_session():
    session = FormFillSession(workflow_key=KEY, steps={"port_id": {"node": "n-asd"}})
    restored = FormFillSession.model_validate_json(session.model_dump_json())
    assert restored.steps == {"port_id": {"node": "n-asd"}}
    assert FormFillSession.model_validate({"workflow_key": KEY}).steps == {}
