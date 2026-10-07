"""A widget field asked in steps — a node first, then a free port on it — the way the frontend's port select works."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.form_fill.widgets import CascadeWidget, Option, Step, WidgetContext
from orchestrator_agent.state import FormFillSession, SearchState

from .test_form_fill import FakeCore, open_form, rejection, turn

KEY = "create_port"
BACK = "Choose another node"  # offered with the ports: back to the node step
PORT_ID: dict[str, Any] = {
    "format": "imsPortId",
    "title": "Port Id",
    "type": "integer",
    "uniforms": {"interfaceSpeed": 10000, "imsPortMode": "patched"},
}
PORT_MODE: dict[str, Any] = {"enum": ["tagged", "untagged"], "title": "Port Mode", "type": "string"}
PAGE = {
    "properties": {"port_id": PORT_ID, "port_mode": PORT_MODE},
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
    assert port.title == "Port Id" and port.values == (101, 102, BACK)
    assert port.choices == ("xe-0/0/1 (free) (10GBASE-LR)", "xe-0/0/2 (free) (10GBASE-LR)", BACK)
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
    assert again.status == "gathering" and again.question("port_id").values == (201, 202, BACK)
    assert all(page.get("port_id") != 101 for page in core.submitted)


def test_the_chosen_steps_are_kept_with_the_session():
    session = FormFillSession(workflow_key=KEY, steps={"port_id": {"node": "n-asd"}})
    restored = FormFillSession.model_validate_json(session.model_dump_json())
    assert restored.steps == {"port_id": {"node": "n-asd"}}
    assert FormFillSession.model_validate({"workflow_key": KEY}).steps == {}


class ManyNodes(NodePorts):
    """A network with more nodes than are read in full: a typed node is searched among them."""

    async def nodes(self, field: Mapping[str, Any], ctx: WidgetContext, chosen: Mapping[str, Any]) -> Sequence[Option]:
        many = [Option(f"n-{n:03d}", f"{n:03d} Node Access lab{n:03d}a-jnx-01") for n in range(250)]
        return [*many, Option("n-asd", "Node Access asd001b-jnx-99", aliases=("asd001b-jnx-99",))]

    async def fetch_chosen(
        self, field: Mapping[str, Any], ctx: WidgetContext, chosen: Mapping[str, Any]
    ) -> Sequence[Option]:
        return PORTS["n-asd"] if chosen["node"] == "n-asd" else []


class Picks:
    """A chooser that picks the first candidate whose label contains the words."""

    def __init__(self) -> None:
        self.offered: list[int] = []

    async def choose(self, title, options, words):
        self.offered.append(len(options))
        return [value for value, label in options if words.casefold() in label.casefold()][:1]


async def test_a_node_typed_loosely_among_many_is_searched_in_the_steps_own_options():
    core, state, chooser = PortCore(), SearchState(), Picks()
    skill = FormFillSkill(widgets=[ManyNodes()], choose=chooser)
    first = await open_form(skill, core, state, KEY)
    assert first.question("port_id").hint == "Type a name or part of it — 251 options."
    ports = await turn(skill, core, state, {"port_id": "asd001b"})  # part of the device name: no exact match
    assert ports.question("port_id").title == "Port Id" and ports.question("port_id").values == (101, 102, BACK)
    assert chooser.offered and chooser.offered[0] <= 50  # read among the candidates the search found, not all


class SinglePort(NodePorts):
    """A network where a node has one free port, or none."""

    async def fetch_chosen(
        self, field: Mapping[str, Any], ctx: WidgetContext, chosen: Mapping[str, Any]
    ) -> Sequence[Option]:
        return {"n-asd": [Option(101, "xe-0/0/1 (free) (10GBASE-LR)")], "n-ut": []}[chosen["node"]]


async def test_a_single_free_port_is_asked_never_taken_for_the_person():
    core, state = PortCore(), SearchState()
    skill = FormFillSkill(widgets=[SinglePort()], choose=NoFit())
    await open_form(skill, core, state, KEY)
    asked = await turn(skill, core, state, {"port_id": "n-asd", "port_mode": "tagged"})
    port = asked.question("port_id")
    assert asked.status == "gathering" and port.title == "Port Id" and port.values == (101, BACK)
    assert all(page.get("port_id") is None for page in core.submitted)  # nothing was picked for the person
    done = await turn(skill, core, state, {"port_id": 101})
    assert done.status == "confirming" and done.values["port_id"] == 101


async def test_a_node_without_free_ports_says_so_and_another_node_can_be_chosen():
    core, state = PortCore(), SearchState()
    skill = FormFillSkill(widgets=[SinglePort()], choose=NoFit())
    await open_form(skill, core, state, KEY)
    empty = await turn(skill, core, state, {"port_id": "n-ut"})
    port = empty.question("port_id")
    assert port.title == "Port Id" and port.values == (BACK,) and "no options" in port.hint.casefold()
    assert BACK in port.hint
    again = await turn(skill, core, state, {"port_id": BACK})  # back to the node
    assert again.question("port_id").title == "Port Id — Node"
    assert state.form_fill.steps.get("port_id", {}) == {}


async def test_a_picked_port_that_is_gone_is_asked_again_not_kept():
    widget = NodePorts()
    core, state = PortCore(), SearchState()
    skill = FormFillSkill(widgets=[widget], choose=NoFit())
    await open_form(skill, core, state, KEY)
    await turn(skill, core, state, {"port_id": "n-asd"})
    PORTS["n-asd"] = [Option(102, "xe-0/0/2 (free) (10GBASE-LR)"), Option(103, "xe-0/0/3 (free) (10GBASE-LR)")]
    try:
        again = await turn(skill, core, state, {"port_id": 101, "port_mode": "tagged"})  # 101 was taken meanwhile
    finally:
        PORTS["n-asd"] = [Option(101, "xe-0/0/1 (free) (10GBASE-LR)"), Option(102, "xe-0/0/2 (free) (10GBASE-LR)")]
    assert again.status == "gathering" and again.question("port_id").values == (102, 103, BACK)
    assert all(page.get("port_id") != 101 for page in core.submitted)


class ImsRejectsCore(PortCore):
    """Core whose port validator rejects a port the free-ports list offered (``Port not found in ims``)."""

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        inputs = args.get("page_inputs") or []
        if name == "get_workflow_form" and inputs and inputs[0].get("port_id") == 101:
            self.submitted.extend(inputs)
            raise ModelRetry(rejection({"port_id": "Port not found in ims"}))
        return await super().__call__(name, args)


async def test_a_picked_port_core_rejects_keeps_the_form_open_with_another_node_possible():
    core, state = ImsRejectsCore(), SearchState()
    skill = FormFillSkill(widgets=[SinglePort()], choose=NoFit())
    await open_form(skill, core, state, KEY)
    await turn(skill, core, state, {"port_id": "n-asd", "port_mode": "tagged"})
    rejected = await turn(skill, core, state, {"port_id": 101})
    port = rejected.question("port_id")
    assert rejected.status == "gathering" and port.problem == "Port not found in ims"
    assert port.values == (101, BACK)
    back = await turn(skill, core, state, {"port_id": BACK})
    assert back.status == "gathering" and back.question("port_id").title == "Port Id — Node"


SERVICE_PAGE = {
    "properties": {
        "ticket_id": {"default": "", "title": "Ticket Id", "type": "string"},
        "port_id": PORT_ID,
        "port_mode": PORT_MODE,
        "lldp": {"default": False, "title": "Lldp", "type": "boolean"},
    },
    "required": ["port_id", "port_mode"],
    "title": "Service Port 10G",
    "type": "object",
}


class ServiceCore(PortCore):
    """Core's form tools for the service-port page: a ticket before the port, its mode and LLDP after it."""

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        if name != "get_workflow_form":
            return await super().__call__(name, args)
        inputs = args["page_inputs"]
        self.submitted.extend(inputs)
        if not inputs:
            return {"page": 0, "complete": False, "schema": SERVICE_PAGE}
        FakeCore.require(SERVICE_PAGE, inputs[0])
        return {"page": 1, "complete": True, "schema": None}


async def test_the_node_ends_its_stop_and_the_port_comes_with_what_follows_it():
    core, state = ServiceCore(), SearchState()
    skill = FormFillSkill(widgets=[NodePorts()], choose=NoFit())
    first = await open_form(skill, core, state, KEY)
    assert set(first.asked) == {"ticket_id", "port_id"}  # up to the node: what follows it depends on the node
    second = await turn(skill, core, state, {"port_id": "n-asd"})  # the ticket left at its default
    assert set(second.asked) == {"port_id", "port_mode", "lldp"} and second.question("port_id").title == "Port Id"
    done = await turn(skill, core, state, {"port_id": 101, "port_mode": "tagged"})  # LLDP left at its default
    assert done.status == "confirming" and done.values == {"port_id": 101, "port_mode": "tagged"}


async def test_a_field_left_at_its_default_is_not_asked_again():
    core, state = ServiceCore(), SearchState()
    skill = FormFillSkill(widgets=[NodePorts()], choose=NoFit())
    await open_form(skill, core, state, KEY)
    await turn(skill, core, state, {"port_id": "n-asd"})
    # The port is answered with words that are no port: only the port is asked again, not the defaults.
    again = await turn(skill, core, state, {"port_id": "nowhere", "port_mode": "tagged"})
    assert again.asked == ["port_id"] and "nowhere" in again.question("port_id").hint
