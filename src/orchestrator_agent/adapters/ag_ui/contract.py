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

"""The AG-UI client's contract, in the agent's OpenAPI document.

A client of the AG-UI route reads more than the routes declare: the artifact reference a tool result
carries in place of its payload (with the chart data an aggregate adds), the metadata of the two stops a
workflow form is walked with, and the answers that resolve them all travel inside AG-UI messages, where no
route names them. They are added to the document's components, so a client generates its types from
``/openapi.json`` alone: the chat popup does (``npm run contract``), from a running agent or from the copy
it checks in, which ``tests/test_openapi.py`` keeps current:

    uv run python -m orchestrator_agent.adapters.ag_ui.contract demos/web/src/api/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaMode, models_json_schema

from orchestrator_agent.adapters.ag_ui.interrupts import ApprovalAnswer, ApprovalStop, AskAnswers, AskStop
from orchestrator_agent.artifacts import DataArtifact, ExportArtifact, QueryArtifact

# What the client reads, as the agent serialises it; what the client sends, as the agent validates it (its
# optional fields optional).
CARRIED: tuple[tuple[type[BaseModel], JsonSchemaMode], ...] = (
    (QueryArtifact, "serialization"),
    (DataArtifact, "serialization"),
    (ExportArtifact, "serialization"),
    (AskStop, "serialization"),
    (ApprovalStop, "serialization"),
    (AskAnswers, "validation"),
    (ApprovalAnswer, "validation"),
)


def openapi_with_the_agui_contract(app: FastAPI) -> dict[str, Any]:
    """The app's OpenAPI document with the models that travel inside AG-UI messages among its schemas."""
    if app.openapi_schema is None:
        document = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        _, carried = models_json_schema(list(CARRIED), ref_template="#/components/schemas/{model}")
        document.setdefault("components", {}).setdefault("schemas", {}).update(carried["$defs"])
        app.openapi_schema = document
    return app.openapi_schema


def main(argv: list[str]) -> None:
    """Write the document to the file named (or stdout), as a running agent serves it at ``/openapi.json``."""
    from orchestrator_agent.app import app

    text = json.dumps(app.openapi(), indent=2) + "\n"
    if argv:
        Path(argv[0]).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv[1:])
