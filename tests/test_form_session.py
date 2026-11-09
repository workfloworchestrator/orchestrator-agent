"""The form-fill state machine: what a turn's response is as an event, what each does, and the one writer."""

import ast
from pathlib import Path

import pytest
from orchestrator.core.schemas.mcp_tools import WorkflowFormPage

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.session import (
    Action,
    Approved,
    FormPhase,
    Message,
    Rejected,
    Shown,
    event_of,
    transition,
)
from orchestrator_agent.state import Decision, FormFillSession, FormInput


@pytest.mark.parametrize(
    ("given", "unseen", "expected"),
    [
        pytest.param(None, False, Message(), id="a-message-is-no-response"),
        pytest.param(None, True, Shown(), id="a-stop-not-shown-yet"),
        pytest.param(FormInput(decision=Decision.CANCEL), False, Rejected(), id="rejected"),
        pytest.param(
            FormInput(decision=Decision.CANCEL, reason="make it 1G"),
            False,
            Rejected("make it 1G"),
            id="rejected-with-words",
        ),
        pytest.param(FormInput(decision=Decision.CANCEL, reason="  "), False, Rejected(), id="blank-words-are-none"),
        pytest.param(FormInput(decision=Decision.START), False, Approved(), id="approved"),
        pytest.param(FormInput(), False, Shown(), id="nothing-readable-shows-the-stop-again"),
    ],
)
def test_what_a_turns_input_is(given, unseen, expected):
    assert event_of(given, unseen) == expected


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (Approved(), Action.START),
        (Rejected(), Action.CANCEL),
        (Rejected("no, 1G"), Action.REOPEN),
        (Shown(), Action.SUMMARY),
        (Message(), Action.REOPEN),
    ],
    ids=lambda x: getattr(x, "name", type(x).__name__),
)
def test_at_the_approval_the_response_decides(event, expected):
    assert transition(FormPhase.CONFIRMING, event) is expected


@pytest.mark.parametrize("phase", [FormPhase.GATHERING, FormPhase.DONE])
@pytest.mark.parametrize("event", [Approved(), Rejected("x"), Shown(), Message()], ids=lambda e: type(e).__name__)
def test_outside_the_approval_every_turn_is_the_models(phase, event):
    assert transition(phase, event) is Action.NOTHING


# --- the writes keep the session consistent ---------------------------------------------------------------------


def _page(index: int, title: str = "T") -> WorkflowFormPage:
    return WorkflowFormPage(page=index, complete=False, status="next", title=title, schema={"type": "object"})


def test_a_form_with_no_page_left_is_complete_and_an_accepted_page_starts_the_next_one_fresh():
    session = FormFillSession(workflow_key="w")
    fs.opened(session, None)
    assert session.status == "confirming" and fs.current(session) is None
    fs.opened(session, _page(0))
    fs.rejected(session, {"a": 1}, [])
    fs.counted(session)
    fs.accepted(session, {"a": 1}, _page(1))
    assert (session.draft, session.rejected, session.calls) == ({}, [], 0)
    fs.restarted(session, 0, None)  # the form has no page there any more
    assert session.status == "confirming" and session.pages == [] and session.page_inputs == []


# --- one writer: the session is written in session.py and nowhere else --------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src" / "orchestrator_agent"
FIELDS = set(FormFillSession.model_fields)
WRITERS = {"update", "pop", "append", "extend", "clear", "setdefault", "insert", "remove"}


def _is_session(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "session") or (
        isinstance(node, ast.Attribute) and node.attr == "form_fill"
    )


def _field_of(node: ast.AST) -> str | None:
    """The session field an expression reaches into (``session.found[...]``, ``session.pages.append``)."""
    while isinstance(node, (ast.Subscript, ast.Call, ast.Attribute)):
        if isinstance(node, ast.Attribute) and node.attr in FIELDS and _is_session(node.value):
            return node.attr
        node = node.func if isinstance(node, ast.Call) else node.value
    return None


def _writes(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in WRITERS:
            targets = [node.func.value]
        line = getattr(node, "lineno", 0)
        found.extend((line, field) for target in targets if (field := _field_of(target)))
    return found


def test_only_the_session_module_writes_the_session():
    offenders = {
        f"{path.relative_to(SRC)}:{line} writes session.{field}"
        for path in SRC.rglob("*.py")
        if path.name != "session.py"
        for line, field in _writes(ast.parse(path.read_text()))
    }
    assert offenders == set()
