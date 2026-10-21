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

"""The OpenAPI document: it carries the AG-UI client's contract, and the popup's checked-in copy is current."""

import json
from pathlib import Path

from orchestrator_agent.app import app

SNAPSHOT = Path(__file__).resolve().parents[1] / "demos" / "web" / "src" / "api" / "openapi.json"

REGENERATE = (
    "the API changed; refresh the popup's copy of the OpenAPI document and its types:\n"
    "  uv run python -m orchestrator_agent.adapters.ag_ui.contract demos/web/src/api/openapi.json\n"
    "  (cd demos/web && npm run contract)\n"
    "or, against a running agent: (cd demos/web && npm run contract:pull)"
)


def test_the_popups_copy_of_the_document_is_current():
    """Every route and schema the app declares at import is in the copy, unchanged.

    A running agent's document has the routes its adapters add at startup (A2A, chat completions) as well;
    the copy may carry those, the popup does not read them.
    """
    copy = json.loads(SNAPSHOT.read_text())
    document = app.openapi()
    assert copy["info"]["title"] == document["info"]["title"]
    for path, operations in document["paths"].items():
        assert copy["paths"].get(path) == operations, f"{path}: {REGENERATE}"
    for name, schema in document["components"]["schemas"].items():
        assert copy["components"]["schemas"].get(name) == schema, f"{name}: {REGENERATE}"


def test_the_document_carries_what_travels_inside_agui_messages():
    schemas = app.openapi()["components"]["schemas"]
    for name in (
        "ChatSummary",
        "Chat",
        "QueryArtifact",
        "ChartData",
        "AskStop",
        "AskedQuestion",
        "ApprovalStop",
        "AskAnswers",
    ):
        assert name in schemas, name
    assert "values" not in schemas["AskedQuestion"]["properties"], (
        "the form values behind the choices stay on the agent"
    )
    assert schemas["QueryArtifact"]["properties"]["chart"]["anyOf"][0] == {"$ref": "#/components/schemas/ChartData"}
