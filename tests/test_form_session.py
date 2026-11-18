"""A form's state: what its writes do, and the one writer."""

import ast
from pathlib import Path

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.state import FormFillSession


def _page(index: int, title: str = "T") -> WorkflowFormPage:
    return WorkflowFormPage(page=index, complete=False, status="next", title=title, schema={"type": "object"})


def test_a_form_with_no_page_left_is_complete_and_an_accepted_page_starts_the_next_one_fresh():
    session = FormFillSession(workflow_key="w")
    fs.opened(session, None)
    assert session.status == "confirming" and fs.current(session) is None
    fs.opened(session, _page(0))
    fs.rejected(session, {"a": 1}, [])
    fs.accepted(session, {"a": 1}, {}, _page(1))
    assert (session.draft, session.rejected) == ({}, [])
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
