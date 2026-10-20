"""The form-fill state machine: every cell of its transition table, and what a turn's input is as an event."""

import pytest

from orchestrator_agent.form_fill.session import (
    TRANSITIONS,
    Action,
    Answered,
    Approved,
    Cancelled,
    Event,
    FormPhase,
    Opened,
    Shown,
    Unanswered,
    event_of,
    transition,
)
from orchestrator_agent.state import Decision, FormInput

EVENTS: tuple[type[Event], ...] = (Opened, Answered, Approved, Cancelled, Unanswered, Shown)
EXPECTED = {
    FormPhase.OPENING: (Action.WALK, *([Action.ABANDON] * 5)),
    FormPhase.GATHERING: (Action.NOTHING, Action.WALK, Action.WALK, Action.CANCEL, Action.ABANDON, Action.WALK),
    FormPhase.CONFIRMING: (Action.NOTHING, Action.WALK, Action.START, Action.CANCEL, Action.ABANDON, Action.WALK),
    FormPhase.DONE: tuple([Action.NOTHING] * 6),
}


def test_the_table_covers_every_phase_and_event_once():
    assert set(TRANSITIONS) == {(phase, event) for phase in FormPhase for event in EVENTS}


@pytest.mark.parametrize("phase", list(FormPhase))
@pytest.mark.parametrize("event", EVENTS, ids=lambda event: event.__name__)
def test_each_transition(phase, event):
    instance = event({"speed": "1000"}) if event is Answered else event()
    assert transition(phase, instance) is EXPECTED[phase][EVENTS.index(event)]


@pytest.mark.parametrize(
    ("given", "unseen", "expected"),
    [
        pytest.param(None, False, Unanswered(), id="a-message-is-no-response"),
        pytest.param(None, True, Shown(), id="a-stop-not-shown-yet"),
        pytest.param(FormInput(decision=Decision.CANCEL), False, Cancelled(), id="rejected"),
        pytest.param(FormInput(values={"a": 1}, decision=Decision.CANCEL), False, Cancelled(), id="a-cancel-wins"),
        pytest.param(FormInput(values={"a": 1}), False, Answered({"a": 1}), id="answers"),
        pytest.param(FormInput(values={}, decision=Decision.START), False, Approved(), id="approved"),
        pytest.param(FormInput(), False, Shown(), id="nothing-readable-shows-the-stop-again"),
    ],
)
def test_what_a_turns_input_is(given, unseen, expected):
    assert event_of(given, unseen) == expected


# --- one writer: the session is written in session.py and nowhere else --------------------------------------

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

from orchestrator_agent.state import FormFillSession  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "orchestrator_agent"
FIELDS = set(FormFillSession.model_fields)
WRITERS = {"update", "pop", "append", "extend", "clear", "setdefault", "insert", "remove"}


def _is_session(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "session") or (
        isinstance(node, ast.Attribute) and node.attr == "form_fill"
    )


def _field_of(node: ast.AST) -> str | None:
    """The session field an expression reaches into (``session.values[...]``, ``session.steps.setdefault(...)[...]``)."""
    while isinstance(node, (ast.Subscript, ast.Call, ast.Attribute)):
        if isinstance(node, ast.Attribute) and node.attr in FIELDS and _is_session(node.value):
            return node.attr
        node = node.func if isinstance(node, ast.Call) else node.value
    return None


def _writes(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in WRITERS:
            targets = [node.func.value]
        found.extend((node.lineno, field) for target in targets if (field := _field_of(target)))
    return found


def test_only_the_session_module_writes_the_session():
    offenders = {
        f"{path.relative_to(SRC)}:{line} writes session.{field}"
        for path in SRC.rglob("*.py")
        if path.name != "session.py"
        for line, field in _writes(ast.parse(path.read_text()))
    }
    assert offenders == set()
