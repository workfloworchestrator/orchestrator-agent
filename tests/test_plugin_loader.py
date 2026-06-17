"""Tests for the frontmatter plugin loader and the PluginSpec model."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from pydantic import ValidationError

from orchestrator_agent.capabilities.loader import load_plugin_specs
from orchestrator_agent.capabilities.spec import PluginSpec


def test_default_specs_present():
    by_id = {s.id: s for s in load_plugin_specs()}
    assert {"search", "aggregate", "entity", "export"} <= set(by_id)


def test_specs_sorted_by_id():
    ids = [s.id for s in load_plugin_specs()]
    assert ids == ["aggregate", "entity", "export", "search"]


def test_spec_round_trips_through_json():
    spec = PluginSpec(id="catalog", description="d", instructions="i", defer_loading=False)
    assert PluginSpec.model_validate_json(spec.model_dump_json()) == spec


def test_frontmatter_rejects_unknown_key():
    with pytest.raises(ValidationError):
        PluginSpec.model_validate({"id": "x", "description": "d", "defer_loading": False, "bogus": 1})


def test_frontmatter_requires_defer_loading():
    with pytest.raises(ValidationError, match="defer_loading"):
        PluginSpec.model_validate({"id": "x", "description": "d"})


def _write_plugin(folder, plugin_id, body, **frontmatter):
    folder.mkdir(parents=True, exist_ok=True)
    frontmatter = {"id": plugin_id, "description": "d", "defer_loading": "false", **frontmatter}
    fm = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    (folder / f"{plugin_id}.md").write_text(f"---\n{fm}\n---\n{body}")


def test_malformed_frontmatter_raises(tmp_path):
    (tmp_path / "broken.md").write_text("no frontmatter here\n")
    with pytest.raises(ValueError, match="frontmatter"):
        load_plugin_specs(tmp_path)


def test_prompts_are_assembled_verbatim_without_substitution(tmp_path):
    # Prompts are plain Markdown now — the body verbatim, no ${...} substitution. A literal ${X} or $
    # passes through untouched (no template engine to choke on it).
    _write_plugin(tmp_path, "x", "Run the search. Cost is $5 and ${NotATemplate}.", tools="[SEARCH_TOOL]")
    text = {s.id: s.instructions for s in load_plugin_specs(tmp_path)}["x"]
    assert text == "Run the search. Cost is $5 and ${NotATemplate}."


def test_unknown_owned_tool_constant_raises():
    # A typo'd `tools:` constant fails loud (DeferredToolGate resolves every spec's tools at startup),
    # not silently as no-ownership.
    from orchestrator_agent.capabilities.behavior import owned_tool_names

    with pytest.raises(KeyError, match="BOGUS_TOOL"):
        owned_tool_names(PluginSpec(id="x", description="d", instructions="i", defer_loading=False, tools=["BOGUS_TOOL"]))


def test_underscore_prefixed_files_are_not_plugins(tmp_path):
    # Files prefixed `_` (e.g. notes, drafts) are never loaded as plugins.
    (tmp_path / "_notes.md").write_text("---\nid: notes\ndescription: d\ndefer_loading: false\n---\n# not a plugin\n")
    _write_plugin(tmp_path, "real", "# real")
    ids = {s.id for s in load_plugin_specs(tmp_path)}
    assert "real" in ids
    assert "notes" not in ids
