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

"""The form-fill skill's caller contract: pydantic-forms schemas in, a JSON object keyed by field name in, text out.

Two things live here and nowhere else: which of the caller's values belong to a page, and how every stop is
rendered. Values are not judged here: they go to core as sent and core's form validation is the only
validation. Reading core's browser-oriented page schema into fields is not the contract's business and
lives in ``core_bridge`` until core's form tool returns fields itself. The words the
contract relies on (``yes`` / ``no`` / ``cancel``, ``true`` / ``false``, ``ACCEPTED``) are defined once at
the top: the replies quote them and the parsers accept them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from orchestrator_agent.form_fill.core_bridge import FORMAT_LONG, FORMAT_UUID, error_detail, option_list
from orchestrator_agent.state import AskField, FormField, FormFillSession, Reply


class FormCommand(StrEnum):
    """The three replies that are not values. Exact tokens: no synonyms, no markup, no punctuation."""

    YES = "yes"  # start the workflow as summarised
    NO = "no"  # do not start it
    CANCEL = "cancel"  # abandon the form


YES, NO, CANCEL = FormCommand.YES.value, FormCommand.NO.value, FormCommand.CANCEL.value
TRUE, FALSE = "true", "false"  # how a boolean field's values are shown; pydantic reads them back
ACCEPT_VALUE = "ACCEPTED"  # the value pydantic-forms' ``Accept`` field takes
# core's name for a subscription reference: the field of its ``ModifySubscriptionPage``, the first page of
# every modify / terminate workflow.
SUBSCRIPTION_ID = "subscription_id"
NOT_OFFERED = "not offered for this subscription"  # a workflow core's per-subscription listing does not have

_FENCE = "```"
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def command(text: str) -> FormCommand | None:
    """The command a reply is, when it is exactly one (surrounding whitespace aside); else None."""
    try:
        return FormCommand(text.strip())
    except ValueError:
        return None


def uuids_in(text: str) -> list[str]:
    """Every UUID mentioned in the text, in order, lower-cased and de-duplicated."""
    return list(dict.fromkeys(m.group(0).lower() for m in _UUID.finditer(text)))


# --- answers <- caller text ---------------------------------------------------------------------


def raw_pairs(text: str) -> dict[str, Any]:
    """The JSON object a reply consists of, keyed as sent, uncoerced.

    A reply that is not a JSON object has no pairs (``{}``): prose is never mined for values. The caller
    may batch values for pages it has not been asked about yet; the skill keeps them pending and applies
    them when the field appears.
    """
    pairs: dict[str, Any] = {}
    for key, raw in (json_object_in(text) or {}).items():
        pairs.setdefault(str(key), raw)
    return pairs


def json_object_in(text: str) -> dict[str, Any] | None:
    """The JSON object the whole message is (a Markdown code fence around it is allowed), else None."""
    body = _unfenced(text)
    if not body.startswith("{"):
        return None
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _unfenced(text: str) -> str:
    """The message without the Markdown code fence a model tends to put around JSON."""
    body = text.strip()
    if body.startswith(_FENCE) and body.endswith(_FENCE) and len(body) >= 2 * len(_FENCE):
        body = body[len(_FENCE) : -len(_FENCE)]
        head, _, rest = body.partition("\n")  # the opening fence may carry a language tag
        body = body if head.lstrip().startswith("{") else rest
    return body.strip()


# --- rendering ------------------------------------------------------------------------------------


def render_pair(name: str, value: str) -> str:
    """One ``name: value`` line of a stop's text (what is filled, what defaults apply)."""
    return f"{name}: {value}"


def label(field: FormField | None, value: Any) -> str:
    """A value as the caller should see it: the option label when there is one, JSON for structured values."""
    if value is None or value == "" or value == []:
        return "(empty)"
    if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)):
        return json.dumps(value)
    if isinstance(value, list):
        return ", ".join(label(field, item) for item in value)
    if field is not None and field.options and str(value) in field.options:
        text = field.options[str(value)]
        return text if text == str(value) else f"{value} ({text})"
    return str(value)


def describe_field(field: FormField) -> str:
    """A field as a stop describes it: title, name, and what it expects (allowed values with labels, shape)."""
    return f"{field.title} (`{field.name}`): {_describe(field)}"


def _describe(field: FormField) -> str:
    match field.kind:
        case "choice":
            return "one of " + option_list(field.options or {})
        case "multi":
            return "one or more of " + option_list(field.options or {})
        case "boolean":
            return f"`{TRUE}` or `{FALSE}`"
        case "accept":
            return f"`{ACCEPT_VALUE}` once the user has approved this step"
        case "integer":
            return "a whole number"
        case "number":
            return "a number"
        case "json":
            return field.shape or "a JSON value"
    if field.format == FORMAT_UUID:
        hint = " of the subscription — find it with the search skill" if field.name == SUBSCRIPTION_ID else ""
        return "an id (UUID)" + hint
    return "free text" + (" (multi-line allowed)" if field.format == FORMAT_LONG else "")


def _default_note(field: FormField) -> str:
    return f", default {label(field, field.default)}" if field.has_default else ""


def filled_so_far(session: FormFillSession, current: Mapping[str, Any] | None = None) -> list[str]:
    values: dict[str, Any] = {k: v for page in session.page_inputs for k, v in page.items()}
    values.update(current or {})
    return [render_pair(name, label(session.fields.get(name), value)) for name, value in values.items()]


def defaults_applying(session: FormFillSession) -> list[str]:
    """Optional fields of the last walk's pages the caller never set: submitted with their form default."""
    given = {k for page in session.page_inputs for k in page}
    return [
        f"{render_pair(f.name, label(f, f.default))} (default)"
        for f in session.fields.values()
        if f.name not in given and f.has_default and not f.display_only
    ]


def need_input(
    session: FormFillSession,
    *,
    page: int,
    title: str | None,
    fields: Sequence[FormField],
    values: Mapping[str, Any],
    missing: Sequence[FormField],
) -> Reply:
    """The stop at a page with values still needed: the contract text, and the same as questions."""
    optional = [
        f for f in fields if not f.required and not f.display_only and f.name not in values and f not in missing
    ]
    lines = [
        f'Form "{title or session.workflow_key}" (workflow `{session.workflow_key}`), page {page} — values needed:'
    ]
    lines += [f"- {f.name} (required): {_describe(f)}" for f in missing]
    lines += [f"- {f.name} (optional{_default_note(f)}): {_describe(f)}" for f in optional]
    if filled := filled_so_far(session, values):
        lines.append("Filled so far: " + "; ".join(filled))
    example = json.dumps({f.name: "..." for f in (missing or optional[:1])})
    lines.append(
        f"Reply with a JSON object keyed by field name, e.g. {example}, or `{CANCEL}` to abandon the form. "
        "Ask the user for anything you don't know."
    )
    ask = [ask_field(f, required=True) for f in missing] + [ask_field(f, required=False) for f in optional]
    return Reply("\n".join(lines), ask=ask)


def ask_field(field: FormField, *, required: bool, problem: str | None = None) -> AskField:
    """The field as one question for a person (chips for its options); ``problem`` is why core rejected the last answer."""
    note = "required" if required else "optional" + _default_note(field) + " — leave empty to keep it"
    question = f"{field.title} (`{field.name}`, {note}): {_describe(field)}"
    if problem:
        question += f" — the last answer was rejected: {problem}"
    match field.kind:
        case "choice" | "multi":
            # The human sees labels; the adapter maps a picked label back to its value.
            options = field.options or {}
            choices, values = tuple(options.values()), tuple(options)
        case "boolean":
            choices = values = (TRUE, FALSE)
        case "accept":
            choices = values = (ACCEPT_VALUE,)
        case _:
            choices = values = ()
    return AskField(name=field.name, question=question, choices=choices, values=values, multiple=field.kind == "multi")


def render_summary(session: FormFillSession) -> str:
    lines = [f"All pages of workflow `{session.workflow_key}` are filled. Values to be submitted:"]
    lines += [f"- {line}" for line in filled_so_far(session)]
    lines += [f"- {line}" for line in defaults_applying(session)]
    lines.append(
        f"Ask the user to confirm. Reply `{YES}` to start the workflow, `{NO}` to cancel, "
        "or send a JSON object with the corrected values."
    )
    return "\n".join(lines)


def render_started(workflow_key: str, process_id: str) -> str:
    return f"Started workflow `{workflow_key}`. Process id: `{process_id}`."


def render_cancelled(workflow_key: str) -> str:
    return f"Cancelled; workflow `{workflow_key}` was not started."


def render_rejected(session: FormFillSession, error: str) -> str:
    return (
        f"The orchestrator rejected the values for workflow `{session.workflow_key}`: {error_detail(error)}\n"
        "Send a JSON object with corrected values for the fields named above."
    )


def render_start_unknown(workflow_key: str) -> str:
    return (
        f"Starting workflow `{workflow_key}` failed before the orchestrator answered; whether a process was "
        "started is unknown. Check the processes before asking to start it again."
    )


def candidate_line(key: str, description: str, reason: str | None) -> str:
    """One workflow on offer; when core cannot run it on the named subscription now, why."""
    return f"`{key}`: {description}" + (f" — cannot run on this subscription now: {reason}" if reason else "")


def render_blocked(workflow_key: str, description: str, reason: str, runnable: Mapping[str, str]) -> str:
    """The handed-off workflow is one core cannot run on the named subscription now: why, and what it can run."""
    lines = [
        "Cannot start a workflow on this subscription now:",
        f"- {candidate_line(workflow_key, description, reason)}",
    ]
    if runnable:
        lines.append("Workflows this subscription can run now:")
        lines += [f"- {candidate_line(key, text, None)}" for key, text in runnable.items()]
    lines.append(
        "Nothing was started. Tell the user why; a new request is needed once the subscription can be worked on."
    )
    return "\n".join(lines)


__all__ = [
    "ACCEPT_VALUE",
    "CANCEL",
    "FALSE",
    "NO",
    "NOT_OFFERED",
    "SUBSCRIPTION_ID",
    "TRUE",
    "YES",
    "FormCommand",
    "ask_field",
    "candidate_line",
    "describe_field",
    "command",
    "filled_so_far",
    "json_object_in",
    "label",
    "need_input",
    "raw_pairs",
    "render_blocked",
    "render_cancelled",
    "render_pair",
    "render_rejected",
    "render_start_unknown",
    "render_started",
    "render_summary",
    "uuids_in",
]
