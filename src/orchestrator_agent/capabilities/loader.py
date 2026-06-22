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

"""Discover and load plugins from disk.

A plugin is a single Markdown file with YAML frontmatter: ``plugins/<id>.md``. The frontmatter
holds metadata (validated by ``PluginSpec``); the body **is** the prompt, used verbatim. Prompts describe
*intent* and do not name MCP tools — the model binds intent to a tool from the tool's own
description (filter/operator usage lives in those descriptions too), the plugin owns the action tool
it needs (declared in ``tools:``), and ``FilterPathGuard`` enforces the discover-before-filter
sequence. Files prefixed ``_`` are never loaded as plugins. The agent-level **system prompt** is
**not** a plugin — it lives at ``capabilities/system_prompt.md`` (a sibling of ``plugins/``) and is
loaded by ``load_system_prompt``.

Plugins are loaded only from the built-in ``plugins/`` directory; to add one, drop an ``<id>.md``
file there (a fork, like any other behaviour change).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from orchestrator_agent.capabilities.spec import PluginSpec
from orchestrator_agent.settings import agent_settings

_BUILTIN_DIR = Path(__file__).parent / "plugins"
# The agent-level system prompt — a sibling of plugins/, NOT a plugin (loaded separately).
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"

# YAML frontmatter delimited by a leading '---' line and a closing '---' line.
_FRONTMATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n?(?P<body>.*)\Z", re.DOTALL)


def _split_frontmatter(path: Path) -> tuple[dict, str]:
    """Return (frontmatter dict, body) for a plugin file, or raise if the frontmatter is malformed."""
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Plugin '{path.name}' must start with a '---' YAML frontmatter block.")
    return yaml.safe_load(match["meta"]) or {}, match["body"].strip()


def _domain_block() -> str:
    """The operator domain-knowledge block, or '' when AGENT_DOMAIN_CONTEXT is unset.

    Injected once into the agent-level system prompt (see ``load_system_prompt``) rather than per-plugin —
    identifier/field conventions are deployment-wide knowledge, not specific to one capability.
    """
    context = agent_settings.AGENT_DOMAIN_CONTEXT.strip()
    return f"## Domain Knowledge\n{context}" if context else ""


def load_plugin_specs(plugins_dir: Path | None = None) -> list[PluginSpec]:
    """Load all plugins from ``plugins_dir`` (default: the built-in dir); the Markdown body is the prompt, verbatim."""
    by_id: dict[str, PluginSpec] = {}
    for path in sorted(p for p in (plugins_dir or _BUILTIN_DIR).glob("*.md") if not p.name.startswith("_")):
        meta, body = _split_frontmatter(path)
        spec = PluginSpec.model_validate(meta)
        spec.instructions = body
        by_id[spec.id] = spec
    return sorted(by_id.values(), key=lambda s: s.id)


def load_system_prompt() -> str:
    """Return the agent-level system prompt (``capabilities/system_prompt.md``) plus the domain block.

    The system prompt is agent-wide — not a plugin — so it lives beside ``plugins/`` and is loaded
    here rather than discovered. ``AGENT_DOMAIN_CONTEXT`` (identifier/field conventions a deployment's
    model can't infer) is appended so it reaches every capability.
    """
    system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    block = _domain_block()
    return f"{system_prompt}\n\n{block}" if block else system_prompt


__all__ = ["load_plugin_specs", "load_system_prompt"]
