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

import csv
import os
import re
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from orchestrator_agent.scratch import SavedFile, Skipped, conversation_folder, rows_of, save_result
from orchestrator_agent.settings import AgentSettings

CONVERSATION = "2D2900FA-E77A-428C-A385-7F46CF16A457"
CANONICAL = "2d2900fa-e77a-428c-a385-7f46cf16a457"


def _grouped(*names: Any, query_id: str | None = "q1") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "results": [{"group_values": {"customer_name": name}, "aggregations": {"count": 1}} for name in names]
    }
    return payload | ({"query_id": query_id} if query_id else {})


def _data_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("conversation", "use_root", "expected"),
    [
        pytest.param(CONVERSATION, True, CANONICAL, id="uuid-is-canonicalised"),
        pytest.param(CANONICAL, True, CANONICAL, id="canonical-uuid-is-kept"),
        pytest.param("abc", True, None, id="not-a-uuid"),
        pytest.param("", True, None, id="empty"),
        pytest.param("../" + CANONICAL, True, None, id="path-traversal"),
        pytest.param(CONVERSATION, False, None, id="no-root"),
    ],
)
def test_conversation_folder(tmp_path: Path, conversation: str, use_root: bool, expected: str | None) -> None:
    folder = conversation_folder(conversation, tmp_path if use_root else None)

    assert folder == (tmp_path / expected if expected else None)
    assert sorted(os.listdir(tmp_path)) == ([expected] if expected else [])


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_conversation_folder_unwritable_root(tmp_path: Path) -> None:
    tmp_path.chmod(0o500)
    try:
        with capture_logs() as logs:
            folder = conversation_folder(CONVERSATION, tmp_path)
    finally:
        tmp_path.chmod(0o700)

    assert folder is None
    assert [log["log_level"] for log in logs] == ["error"]


@pytest.mark.parametrize(
    ("tool_name", "payload", "expected"),
    [
        pytest.param(
            "aggregate",
            {"results": [{"group_values": {"customer_name": "A"}, "aggregations": {"count": 2}}]},
            (("customer_name", "count"), [("A", 2)]),
            id="grouped-aggregate",
        ),
        pytest.param(
            "aggregate",
            {"results": [{"group_values": {"customer_name": None}, "aggregations": {"count": 2}}]},
            (("customer_name", "count"), [("", 2)]),
            id="none-group-value-is-empty",
        ),
        pytest.param(
            "aggregate",
            {
                "results": [
                    {"group_values": {"customer_name": "A"}, "aggregations": {"count": 2}},
                    {"group_values": {"customer_name": "B", "site": "X"}, "aggregations": {"sum_bw": 5}},
                ]
            },
            (("customer_name", "site", "count", "sum_bw"), [("A", "", 2, ""), ("B", "X", "", 5)]),
            id="missing-keys-keep-columns-consistent",
        ),
        pytest.param(
            "aggregate",
            {"results": [{"group_values": {}, "aggregations": {"count": 2}}]},
            None,
            id="scalar-aggregate",
        ),
        pytest.param("aggregate", {"results": []}, None, id="zero-rows"),
        pytest.param("aggregate", "not a payload", None, id="not-a-mapping"),
        pytest.param(
            "search",
            {"results": [{"entity_id": "e1", "entity_type": "SUBSCRIPTION", "title": "T", "score": 0.5}]},
            (("entity_id", "entity_type", "title", "score"), [("e1", "SUBSCRIPTION", "T", 0.5)]),
            id="search-rows",
        ),
        pytest.param("search", {"results": []}, None, id="search-without-rows"),
        pytest.param("resolve_entity", {"results": [{"entity_id": "e1"}]}, None, id="other-tool"),
    ],
)
def test_rows_of_payload(tool_name: str, payload: Any, expected: Any) -> None:
    assert rows_of(tool_name, payload) == expected


def test_csv_round_trips_awkward_values(tmp_path: Path) -> None:
    names = ["Coöperatieve Vereniging, u.a.", 'Say "hi"', "two\nlines", "GÉANT"]

    saved = save_result(tmp_path, "aggregate", _grouped(*names), max_bytes=1_000_000)

    assert isinstance(saved, SavedFile)
    assert not (tmp_path / saved.name).read_bytes().startswith(b"\xef\xbb\xbf")
    assert [row["customer_name"] for row in _data_rows(tmp_path / saved.name)] == names


def test_saving_twice_replaces_atomically(tmp_path: Path) -> None:
    save_result(tmp_path, "aggregate", _grouped("A", "B"), max_bytes=1_000_000)
    saved = save_result(tmp_path, "aggregate", _grouped("A", "B", "C"), max_bytes=1_000_000)

    assert isinstance(saved, SavedFile)
    assert saved.rows == 3
    assert len(_data_rows(tmp_path / "q-q1.csv")) == 3
    assert sorted(os.listdir(tmp_path)) == ["q-q1.csv"]


def test_saved_file_describes_the_csv(tmp_path: Path) -> None:
    saved = save_result(tmp_path, "aggregate", _grouped("A"), max_bytes=1_000_000)

    assert saved == SavedFile(name="q-q1.csv", rows=1, columns=("customer_name", "count"))


@pytest.mark.parametrize(
    ("query_id", "pattern"),
    [
        pytest.param("abc", r"^q-abc\.csv$", id="query-id"),
        pytest.param(None, r"^aggregate-[0-9a-f]{8}\.csv$", id="no-query-id"),
        pytest.param("../../etc/x", r"^aggregate-[0-9a-f]{8}\.csv$", id="unsafe-query-id"),
    ],
)
def test_file_names(tmp_path: Path, query_id: str | None, pattern: str) -> None:
    saved = save_result(tmp_path, "aggregate", _grouped("A", query_id=query_id), max_bytes=1_000_000)

    assert isinstance(saved, SavedFile)
    assert re.match(pattern, saved.name)
    assert os.listdir(tmp_path) == [saved.name]


def test_too_large_is_skipped(tmp_path: Path) -> None:
    saved = save_result(tmp_path, "aggregate", _grouped("A", "B", "C"), max_bytes=10)

    assert saved == Skipped(rows=3)
    assert os.listdir(tmp_path) == []


def test_not_a_row_result_saves_nothing(tmp_path: Path) -> None:
    assert save_result(tmp_path, "resolve_entity", {"results": [{"a": 1}]}, max_bytes=1_000_000) is None
    assert os.listdir(tmp_path) == []


def test_failed_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    class _Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    payload = {"results": [{"group_values": {"k": _Unprintable()}, "aggregations": {"count": 1}}]}

    with pytest.raises(RuntimeError):
        save_result(tmp_path, "aggregate", payload, max_bytes=1_000_000)

    assert os.listdir(tmp_path) == []


def test_settings_defaults() -> None:
    fields = AgentSettings.model_fields

    assert [
        fields[name].default
        for name in (
            "SCRATCH_DIR",
            "SCRATCH_MAX_FILE_BYTES",
            "SCRATCH_VIRTUAL_DIR",
            "SCRATCH_REPLY_TABLE_MAX_ROWS",
            "SCRATCH_REPLY_PREVIEW_ROWS",
        )
    ] == [None, 20_000_000, "/data", 100, 20]
