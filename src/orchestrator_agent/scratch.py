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

"""A per-conversation scratch folder with the rows of a tool result as CSV files.

An agent calling this one over A2A can read the rows of a ``search`` or ``aggregate`` result by file, instead of
having the planning model retype them. The folder is ``<SCRATCH_DIR>/<conversation id>/`` on a shared volume; the
id is always the canonical form of a UUID (lowercase, hyphenated), and anything else means no folder, so a
conversation id can never point outside the root.

Nothing here runs unless ``SCRATCH_DIR`` is set: the callers pass ``root=None`` and get ``None`` back.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import structlog

from orchestrator_agent.rendering.tables import _ordered_keys

logger = structlog.get_logger(__name__)

SEARCH_COLUMNS = ("entity_id", "entity_type", "title", "score")
_SAFE_QUERY_ID = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class SavedFile:
    """A CSV written to the scratch folder."""

    name: str
    rows: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Skipped:
    """A result with ``rows`` rows that is too large to save."""

    rows: int


def conversation_folder(conversation: str, root: Path | None) -> Path | None:
    """The folder of ``conversation`` under ``root``, created when missing; ``None`` when there is no scratch.

    That is: ``root`` is unset, ``conversation`` is not a UUID, or the folder cannot be created (logged).
    """
    if root is None:
        return None
    try:
        folder = root / str(uuid.UUID(conversation))
    except ValueError:
        return None
    try:
        folder.mkdir(exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create the scratch folder", folder=str(folder), error=str(exc))
        return None
    return folder


def rows_of(tool_name: str, payload: Any) -> tuple[tuple[str, ...], list[tuple[Any, ...]]] | None:
    """The columns and rows of a ``search`` or grouped ``aggregate`` payload; ``None`` for anything else.

    The columns of an aggregate are the grouping keys, then the aggregations, in the order of the Markdown table.
    A key a row lacks, or a ``None`` value, is an empty cell. A scalar aggregate (no ``group_values``), a payload
    without rows and every other tool carry no rows.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    rows = [row for row in results if isinstance(row, dict)] if isinstance(results, list) else []
    match tool_name:
        case "aggregate" if any(row.get("group_values") for row in rows):
            group_keys = _ordered_keys(row.get("group_values") or {} for row in rows)
            metric_keys = _ordered_keys(row.get("aggregations") or {} for row in rows)
            return tuple(group_keys + metric_keys), [
                tuple(
                    _cell(source.get(key))
                    for source, keys in (
                        (row.get("group_values") or {}, group_keys),
                        (row.get("aggregations") or {}, metric_keys),
                    )
                    for key in keys
                )
                for row in rows
            ]
        case "search" if rows:
            return SEARCH_COLUMNS, [tuple(_cell(row.get(column)) for column in SEARCH_COLUMNS) for row in rows]
        case _:
            return None


def save_result(folder: Path, tool_name: str, payload: Any, max_bytes: int) -> SavedFile | Skipped | None:
    """Write the rows of a tool result to ``folder`` as a CSV file.

    ``None`` when the payload carries no rows, ``Skipped`` when the CSV is larger than ``max_bytes`` (nothing is
    written then). The file is complete or absent: it is written to a temporary file in ``folder`` and moved into
    place, so a second save of the same query replaces the first and a reader never sees a half-written file.
    """
    table = rows_of(tool_name, payload)
    if table is None:
        return None
    columns, rows = table
    name = _file_name(tool_name, payload)
    with tempfile.NamedTemporaryFile(
        "w", dir=folder, suffix=".tmp", delete=False, encoding="utf-8", newline=""
    ) as temporary:
        try:
            csv.writer(temporary, lineterminator="\n").writerows(chain([columns], rows))
            temporary.flush()
            too_large = os.path.getsize(temporary.name) > max_bytes
            if not too_large:
                os.replace(temporary.name, folder / name)
        except BaseException:
            Path(temporary.name).unlink(missing_ok=True)
            raise
    if too_large:
        Path(temporary.name).unlink(missing_ok=True)
        return Skipped(rows=len(rows))
    return SavedFile(name=name, rows=len(rows), columns=columns)


def _file_name(tool_name: str, payload: Any) -> str:
    """``q-<query_id>.csv``, or ``<tool>-<random>.csv`` without a usable query id.

    The query id comes from a tool result and becomes part of a path, so only a plain token is used.
    """
    query_id = payload.get("query_id") if isinstance(payload, dict) else None
    match query_id:
        case str() if _SAFE_QUERY_ID.fullmatch(query_id):
            return f"q-{query_id}.csv"
        case _:
            return f"{tool_name}-{uuid.uuid4().hex[:8]}.csv"


def _cell(value: Any) -> Any:
    """A CSV cell: ``None`` is an empty cell, anything else is written as it is."""
    return "" if value is None else value


__all__ = ["SavedFile", "Skipped", "conversation_folder", "rows_of", "save_result"]
