"""Tests for state models — SearchState."""

from __future__ import annotations

import os
from uuid import uuid4

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.state import FormFillSession, SearchState


class TestSearchState:
    def test_defaults(self):
        state = SearchState()
        assert state.user_input == ""
        assert state.run_id is None
        assert state.query_id is None
        assert state.query is None
        assert state.pending_filters is None
        assert state.form_fill is None

    def test_with_values(self):
        rid = uuid4()
        qid = uuid4()
        state = SearchState(user_input="test", run_id=rid, query_id=qid)
        assert state.user_input == "test"
        assert state.run_id == rid
        assert state.query_id == qid

    def test_serialization_roundtrip(self):
        state = SearchState(user_input="test query")
        data = state.model_dump(mode="json")
        restored = SearchState.model_validate(data)
        assert restored.user_input == "test query"

    def test_form_fill_session_roundtrip(self):
        # The validated form pages survive the per-thread snapshot (confirmation happens on a later turn).
        session = FormFillSession(workflow_key="create_node", page_inputs=[{"product": "p-1"}, {}], status="confirming")
        restored = SearchState.model_validate(SearchState(form_fill=session).model_dump(mode="json"))
        assert restored.form_fill == session
