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

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from orchestrator.core.search.core.types import EntityType

from orchestrator_agent.entity_lookup import (
    _ENTITY_LOOKUP,
    IdForm,
    _classify_id,
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
