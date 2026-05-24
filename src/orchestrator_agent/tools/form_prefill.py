# Copyright 2019-2025 SURF, GÉANT.
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

"""LLM-driven pre-fill for workflow form pages.

Given the user's natural-language ask and the current page's JSON Schema,
extract the most likely values for each field. Anything the LLM isn't
confident about is left empty so the user fills it in manually.

This is the bit that makes the multi-page form-fill feel agentic instead of
"a multi-page wizard with extra steps". Without pre-fill the LLM only picks
which tool to call; with it, the LLM actually translates intent into form
values.

Failure is non-fatal: any error returns an empty dict and the form renders
with schema defaults only.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

import structlog
from pydantic_ai import Agent

from orchestrator_agent.settings import agent_settings

logger = structlog.get_logger(__name__)


_PREFILL_INSTRUCTIONS = dedent(
    """
    You pre-fill a workflow form for a user based on their natural-language request.

    Inputs:
    - The form's JSON Schema (one page of a multi-page form).
    - The user's free-text request.
    - Optional: values already submitted on previous pages of the same form.

    What to fill in
    - Values the user stated explicitly.
    - Values that can be directly inferred from the user's text by applying
      conventional, unambiguous mappings a competent reader would make without
      hesitation. The user is reviewing your output and can edit anything,
      so a well-supported inference is more useful than an empty field.
    - For each field, use the JSON Schema's ``title`` / property name / type
      to decide which part of the user's text is relevant; map the user's
      phrasing to a value of the right type.
    - If the schema declares a default and the user said nothing relevant,
      keep the default rather than emitting null.

    What NOT to fill in
    - Values the user never mentioned and cannot be derived from anything they
      did mention. Don't speculate to fill space.
    - For fields with a standard JSON Schema ``format`` such as ``email`` or
      ``uuid``, only fill in if the user's text actually contains a value that
      looks like the right format. Custom (non-standard) formats are treated
      as a suggestion, not a strict requirement.

    Output
    - Strictly JSON: a single object whose keys match the schema's property
      names. No comments, no markdown, no code fences. Use null for fields
      you genuinely can't fill.
    """
).strip()


async def prefill_form_fields(
    user_input: str,
    schema: dict[str, Any] | None,
    previous_pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a ``{field_name: value}`` dict for fields the LLM could fill.

    Fields the LLM can't confidently extract are omitted (null values are
    filtered out). On any error the function returns ``{}`` — the form will
    render with schema defaults only.
    """
    if not schema:
        return {}
    properties = schema.get("properties") or {}
    if not properties:
        return {}
    if not user_input:
        return {}

    prompt = _build_prompt(user_input, schema, previous_pages or [])

    try:
        agent: Agent[None, str] = Agent(model=agent_settings.create_model(), output_type=str)
        result = await agent.run(prompt)
        raw = (result.output or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Form prefill returned non-JSON; skipping", error=str(exc), raw=raw[:200] if "raw" in dir() else ""
        )
        return {}
    except Exception as exc:
        logger.warning("Form prefill failed; skipping", error=str(exc))
        return {}

    if not isinstance(parsed, dict):
        return {}

    # Drop nulls and any keys that aren't in the current page's schema.
    cleaned = {k: v for k, v in parsed.items() if k in properties and v is not None}
    if cleaned:
        logger.debug("Form prefill produced values", count=len(cleaned), keys=list(cleaned.keys()))
    return cleaned


def _build_prompt(user_input: str, schema: dict[str, Any], previous_pages: list[dict[str, Any]]) -> str:
    schema_json = json.dumps(schema, indent=2)
    prev_json = json.dumps(previous_pages, indent=2) if previous_pages else "(none)"
    return dedent(
        f"""
        {_PREFILL_INSTRUCTIONS}

        ## Current page schema
        ```json
        {schema_json}
        ```

        ## User request
        {user_input!r}

        ## Previously submitted pages
        {prev_json}

        Now return the JSON object.
        """
    ).strip()
