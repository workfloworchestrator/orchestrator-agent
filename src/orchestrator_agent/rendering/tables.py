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

"""Deterministic Markdown tables for search results.

The mirror of :mod:`charts`: a search returns structured rows
(``SearchToolResponse.results`` — ``{entity_id, entity_type, title, score}``), so the row
listing is rendered deterministically instead of letting the model hand-format it. This
guarantees the row cap, an accurate "showing N of M", and that no row is silently dropped.

The MCP result is intentionally lightweight (title/type/id/score only), so the columns are
``Title`` and ``ID`` (plus ``Type`` when a search spans multiple entity types). The internal
relevance ``score`` is omitted as noise.
"""

from __future__ import annotations

from typing import Any

ROW_CAP = 10


def search_to_markdown(payload: Any) -> str | None:
    """Render a ``search`` tool result as a Markdown table, or ``None`` when there are no rows."""
    if not isinstance(payload, dict):
        return None
    rows = [row for row in (payload.get("results") or []) if isinstance(row, dict)]
    if not rows:
        return None

    show_type = len({str(row.get("entity_type")) for row in rows if row.get("entity_type")}) > 1
    headers = (["Type"] if show_type else []) + ["Title", "ID"]
    shown = rows[:ROW_CAP]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in shown:
        cells = ([_cell(row.get("entity_type"))] if show_type else []) + [
            _cell(row.get("title")),
            _cell(row.get("entity_id")),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    if len(rows) > ROW_CAP:
        return f"{table}\n\n_Showing {ROW_CAP} of {len(rows)}._"
    if payload.get("has_more"):
        return f"{table}\n\n_Showing {len(rows)}; more available — narrow the search or raise the limit._"
    return table


def _cell(value: Any) -> str:
    """Markdown table cell: stringify and neutralise pipes/newlines that would break the row."""
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


__all__ = ["search_to_markdown"]
