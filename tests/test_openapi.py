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

"""The OpenAPI document carries the AG-UI client's contract: the chat popup generates its types from it."""

from orchestrator_agent.app import app


def test_the_document_carries_what_travels_inside_agui_messages():
    schemas = app.openapi()["components"]["schemas"]
    for name in (
        "ChatSummary",
        "Chat",
        "QueryArtifact",
        "ExportArtifact",
        "ChartData",
        "ApprovalStop",
        "ApprovalAnswer",
    ):
        assert name in schemas, name
    # The form's pages are filled in conversation: no question card, no options route.
    assert not {"AskStop", "AskedQuestion", "AskAnswers", "FieldOption"} & set(schemas)
    assert "/chats/{thread_id}/options/{field}" not in app.openapi()["paths"]
    assert schemas["ApprovalAnswer"]["properties"]["reason"]["anyOf"][0] == {"type": "string"}
    assert schemas["QueryArtifact"]["properties"]["chart"]["anyOf"][0] == {"$ref": "#/components/schemas/ChartData"}
