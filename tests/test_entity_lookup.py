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

"""Tests for the entity_lookup module.

Real prefix-matching SQL requires a live database and is out of scope here; the
resolver test uses a fake session that records the query arguments and returns
canned rows.
"""

from __future__ import annotations

import os
from uuid import UUID

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from orchestrator.core.search.core.types import EntityType

from orchestrator_agent.entity_lookup import (
    _ENTITY_LOOKUP,
    IdForm,
    ResolvedEntity,
    _classify_id,
    resolve_entity_id_prefix,
)


def test_entity_lookup_covers_all_entity_types():
    for entity_type in EntityType:
        assert entity_type in _ENTITY_LOOKUP, f"missing lookup spec for {entity_type}"


class TestClassifyId:
    @pytest.mark.parametrize(
        "raw, expected_form, expected_norm",
        [
            pytest.param(
                "12345678-1234-5678-1234-567812345678",
                IdForm.FULL_UUID,
                "12345678-1234-5678-1234-567812345678",
                id="full-uuid",
            ),
            pytest.param(
                "12345678-1234-5678-1234-567812345678".upper(),
                IdForm.FULL_UUID,
                "12345678-1234-5678-1234-567812345678",
                id="full-uuid-uppercase-normalized",
            ),
            pytest.param("  abcd1234  ", IdForm.PREFIX, "abcd1234", id="prefix-stripped"),
            pytest.param("ABCD1234", IdForm.PREFIX, "abcd1234", id="prefix-lowercased"),
            pytest.param("abcd", IdForm.PREFIX, "abcd", id="prefix-min-length"),
            pytest.param("abc", IdForm.TOO_SHORT, "abc", id="too-short-3-hex"),
            pytest.param("ab-c", IdForm.TOO_SHORT, "ab-c", id="too-short-hyphen-ignored"),
            pytest.param("acme", IdForm.NON_HEX, "acme", id="non-hex-letter"),
            pytest.param("hello world", IdForm.NON_HEX, "hello world", id="non-hex-space"),
            pytest.param("", IdForm.NON_HEX, "", id="empty"),
        ],
    )
    def test_classify(self, raw, expected_form, expected_norm):
        form, norm = _classify_id(raw)
        assert form is expected_form
        assert norm == expected_norm


class _FakeQuery:
    """Records filter/limit calls and returns canned rows from .all()."""

    def __init__(self, recorder: dict, rows: list):
        self._recorder = recorder
        self._rows = rows

    def filter(self, *args):
        self._recorder["filter_called"] = True
        return self

    def limit(self, n):
        self._recorder["limit_arg"] = n
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Stand-in for WrappedSession that records query() column args."""

    def __init__(self, rows: list):
        self.rows = rows
        self.recorder: dict = {}

    def query(self, *args):
        self.recorder["query_args"] = args
        return _FakeQuery(self.recorder, self.rows)


class TestResolveEntityIdPrefix:
    def test_query_machinery(self):
        """Checks that the correct columns, filter, and limit+1 are used."""
        session = _FakeSession([("11111111-aaaa", "Acme Corp")])

        resolve_entity_id_prefix(session, EntityType.SUBSCRIPTION, "1111", limit=10)

        # Columns selected match the spec for this entity type (identity, not ==,
        # because SQLAlchemy columns overload __eq__ to build SQL expressions).
        spec = _ENTITY_LOOKUP[EntityType.SUBSCRIPTION]
        assert session.recorder["query_args"][0] is spec.id_col
        assert session.recorder["query_args"][1] is spec.title_expr
        # limit+1 so the caller can detect "more than limit" without a count query.
        assert session.recorder["limit_arg"] == 11
        assert session.recorder["filter_called"] is True

    @pytest.mark.parametrize(
        "raw_row, expected",
        [
            pytest.param(
                ("11111111-aaaa", "Acme Corp"),
                ResolvedEntity(entity_id="11111111-aaaa", title="Acme Corp"),
                id="first-row",
            ),
            pytest.param(
                ("11111111-bbbb", "Beta LLC"),
                ResolvedEntity(entity_id="11111111-bbbb", title="Beta LLC"),
                id="second-row",
            ),
            pytest.param(
                (UUID("11111111-1111-1111-1111-111111111111"), "Gamma Inc"),
                ResolvedEntity(entity_id="11111111-1111-1111-1111-111111111111", title="Gamma Inc"),
                id="uuid-id-coerced-to-str",
            ),
        ],
    )
    def test_row_mapping(self, raw_row, expected):
        """Each DB row is mapped to a ResolvedEntity with str id and title."""
        session = _FakeSession([raw_row])
        result = resolve_entity_id_prefix(session, EntityType.SUBSCRIPTION, "1111", limit=10)
        assert result == [expected]

    def test_empty_rows_returns_empty_list(self):
        session = _FakeSession([])
        result = resolve_entity_id_prefix(session, EntityType.PRODUCT, "dead", limit=10)
        assert result == []
