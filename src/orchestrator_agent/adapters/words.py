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

"""What a person reads of a form-fill reply, for the transports whose client shows it to one in a chat.

The skill's own text is one JSON object (``FormReply``), written for a calling agent. A person in a chat
(LibreChat, the AG-UI chat popup) gets the same data as a line of Markdown: which form is being filled, at
the approval a table of the values as they will be sent (always, as that is what is approved) with the
workflow's own summary under it, and the outcome once the form is over.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from orchestrator_agent.state import FormReply, Reply, SummaryTable

SUMMARY_CAPTION = "The workflow's own summary:"


def shown(reply: Reply) -> str:
    """A reply as the person reads it: the text above the card of a stop, or the outcome when the form is over.

    The skill's own text is one JSON object, written for a calling agent; a person in a chat gets the same
    data as a line of markdown, and at the approval as a table of the values as they will be sent — always,
    as that is what is approved — with the workflow's own summary under it when its form has one.
    """
    form = _form(reply)
    if form is None:
        return reply.text
    key = f"`{form.workflow_key}`"
    if form.status == "gathering":
        return f"**Workflow form {key}**" + (f" — {form.title}" if form.title else "")
    if form.status == "confirming":
        tables = [line for table in form.summary for line in _summary_table(table)]
        summary = ["", SUMMARY_CAPTION, *tables] if tables else []
        return "\n".join([f"**Start workflow {key} with these values?**", *_values_table(form), *summary])
    if form.status == "started":
        outcome = f"Process id: `{form.process_id}`" if form.process_id else (form.reason or "")
        return f"Workflow {key} started. {outcome}".rstrip()
    if form.status == "cancelled":
        return f"The {key} form was cancelled; nothing was started."
    return f"The {key} form failed: {form.reason}"


def _summary_table(table: SummaryTable) -> list[str]:
    """One table of the workflow's own summary: a row per label, a column per item, headed when the form heads them."""
    columns = table.get("columns") or [[]]
    headers = [*table.get("headers", []), *[""] * len(columns)][: len(columns)]
    lines = ["", "| | " + " | ".join(_cell(header) for header in headers) + " |", "|---|" + "---|" * len(columns)]
    for row, label in enumerate(table.get("labels", [])):
        cells = [_cell(column[row]) if row < len(column) else "" for column in columns]
        lines.append(f"| {_cell(label)} | " + " | ".join(cells) + " |")
    return lines


def _values_table(form: FormReply) -> list[str]:
    """The values as they will be submitted (an id as the label the form has for it), and the defaults that apply."""
    lines: list[str] = []
    if form.values:
        lines += ["", "| Field | Value |", "|---|---|"]
        lines += [f"| {name} | {_cell(form.labels.get(name, value))} |" for name, value in form.values.items()]
    if form.defaults:
        defaults = ", ".join(f"{name} = {_cell(value)}" for name, value in form.defaults.items())
        lines += ["", f"Defaults that apply: {defaults}"]
    return lines


def _form(reply: Reply) -> FormReply | None:
    try:
        return FormReply.model_validate_json(reply.text)
    except ValidationError:
        return None


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")


__all__ = ["SUMMARY_CAPTION", "shown"]
