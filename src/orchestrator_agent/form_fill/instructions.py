# Copyright 2019-2026 SURF, GÉANT.
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

"""What the model is told while a form is open: the rules of filling it, and where the form stands.

Rendered per model request from the session, so the model always reads the current page, the pages
accepted with their values, core's last verdict and the searches so far. The rules are short and
generic: the page's fields are in the ``submit_page`` tool's schema, not here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.model import fields_of
from orchestrator_agent.form_fill.pages import fillable, labels
from orchestrator_agent.state import FormFillSession

RULES = """## Filling the workflow form `{key}`
You are filling the input form of this workflow with the person, page by page. Code keeps the pages; you fill the current one with the page tools.
- The current page's fields are the parameters of `submit_page`; its schema says what each takes. Fill it from what the person said, in this message or earlier, translated into the values the schema takes. Never invent a value.
- A field with a short option list: pick the value whose label is what the person means. A field whose description says to search: call `search_options` with their words and give only a value it returned. When the words fit several options, or none, ask the person which, naming the options; never pick for them.
- Submit as soon as every required field is known; leave an optional field out unless the person said something about it. When the orchestrator rejects the page, fix what it names from what the person said, or ask.
- Ask for everything still missing on the page in one short message, in plain words, with the choices when there are a few. Do not ask again for what was already said, and do not explain the form's mechanics.
- A change to an earlier page: `restart_from_page`, then submit it again. A person who no longer wants the workflow, or asks for something else: `cancel_form` first.
- When `submit_page` answers 'complete', write nothing: the person is shown the values and approves the start themselves. Never say that something was started; only the start's own message says that."""

NO_APPROVAL = "- This client cannot approve a start: the form can be prepared here, not started. Say so if the person asks to start."


def render(session: FormFillSession, *, hitl: bool) -> str:
    """The instructions for this request: the rules, then where the form stands."""
    lines = [RULES.format(key=session.workflow_key)]
    if not hitl:
        lines.append(NO_APPROVAL)
    lines.append("")
    lines.append("### Where the form stands")
    lines.extend(_accepted(session))
    lines.extend(_current(session))
    lines.extend(_found(session))
    if session.reason:
        lines.append(f"The person did not approve the start and said: {json.dumps(session.reason)}. Act on it.")
    return "\n".join(lines)


def _accepted(session: FormFillSession) -> list[str]:
    if not session.page_inputs:
        return ["Accepted pages: none yet."]
    lines = ["Accepted pages:"]
    for index, values in enumerate(session.page_inputs):
        title = session.pages[index].title if index < len(session.pages) and session.pages[index].title else ""
        lines.append(f"- page {index}{f' ({title})' if title else ''}: {_values(session, values)}")
    return lines


def _current(session: FormFillSession) -> list[str]:
    page = fs.current(session)
    if page is None:
        return [
            "The form is complete: every page is accepted. The person wrote instead of approving the start: read it "
            "as a correction (`restart_from_page`, then submit again) or a question; the start is offered again once "
            "the pages are complete."
        ]
    fields = [field for field in fields_of(page) if fillable(field)]
    required = [field.name for field in fields if field.required]
    optional = [field.name for field in fields if not field.required]
    title = f" ({page.title})" if page.title else ""
    lines = [
        f"Current page: {page.page}{title}. Required: {', '.join(required) or 'none'}. Optional: {', '.join(optional) or 'none'}."
    ]
    if session.draft:
        lines.append(f"Last submitted for it: {_values(session, session.draft)}")
    if session.rejected:
        said = "; ".join(f"{'.'.join(str(p) for p in error.loc) or 'page'}: {error.msg}" for error in session.rejected)
        lines.append(f"The orchestrator rejected that: {said}")
    return lines


def _found(session: FormFillSession) -> list[str]:
    if not session.found:
        return []
    lines = ["Options found so far:"]
    for field, options in session.found.items():
        shown = ", ".join(
            f"{value} ({label})" if label != value else value for value, label in list(options.items())[:25]
        )
        lines.append(f"- {field}: {shown}")
    return lines


def _values(session: FormFillSession, values: Mapping[str, Any]) -> str:
    shown = labels(session, values)
    parts = []
    for name, value in values.items():
        text = json.dumps(value, default=str)
        parts.append(
            f"{name} = {text} ({json.dumps(shown[name], default=str)})" if name in shown else f"{name} = {text}"
        )
    return ", ".join(parts) or "nothing"


__all__ = ["NO_APPROVAL", "RULES", "render"]
