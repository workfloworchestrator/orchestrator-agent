"""Tests for artifact models."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.artifacts import DataArtifact, ExportArtifact, QueryArtifact, ToolArtifact


class TestToolArtifact:
    def test_base(self):
        a = ToolArtifact(description="test")
        assert a.description == "test"

    def test_isinstance_check(self):
        q = QueryArtifact(description="test", query_id="q1", total_results=5)
        assert isinstance(q, ToolArtifact)
        d = DataArtifact(description="test", entity_id="e1", entity_type="subscription")
        assert isinstance(d, ToolArtifact)
        e = ExportArtifact(description="test", query_id="q1", download_url="http://example.com/export")
        assert isinstance(e, ToolArtifact)


class TestQueryArtifact:
    def test_serialization(self):
        a = QueryArtifact(description="Found 5", query_id="q1", total_results=5)
        data = a.model_dump()
        assert data["query_id"] == "q1"
        assert data["total_results"] == 5

    def test_default_visualization_type(self):
        a = QueryArtifact(description="test", query_id="q1", total_results=0)
        assert a.visualization_type is not None


class TestDataArtifact:
    def test_fields(self):
        a = DataArtifact(description="Details", entity_id="abc-123", entity_type="subscription")
        assert a.entity_id == "abc-123"
        assert a.entity_type == "subscription"


class TestExportArtifact:
    def test_fields(self):
        a = ExportArtifact(description="Export", query_id="q1", download_url="http://example.com/dl")
        assert a.download_url == "http://example.com/dl"
