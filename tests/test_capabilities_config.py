"""Tests for capability specs."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.capabilities.config import CapabilitySpec, load_capability_specs


def test_default_specs_present_and_always_on():
    by_id = {s.id: s for s in load_capability_specs()}
    assert {"search", "aggregate", "entity", "export"} <= set(by_id)
    assert all(not spec.defer_loading for spec in by_id.values())


def test_spec_round_trips_through_json():
    spec = CapabilitySpec(id="catalog", description="d", instructions="i", defer_loading=True)
    assert CapabilitySpec.model_validate_json(spec.model_dump_json()) == spec
