# Plugin-driven capabilities

The agent's domain capabilities are **plugins**: authored Markdown files an operator can drop in to
extend the agent, with YAML frontmatter for metadata. There is no bespoke template language.

## Vocabulary

| Term | What it is |
|---|---|
| **Plugin** | A single `plugins/<id>.md` file: YAML frontmatter + a Markdown body. |
| **`PluginSpec`** | One model: validates the frontmatter (`extra="forbid"` — typos fail loudly) and carries the body verbatim as `instructions`. |
| pydantic-ai capability | Runtime projection — instructions (+ optional behaviour) surfaced to the model. |
| A2A `AgentSkill` | Runtime projection — advertised catalog entry (when `advertise: true`). |

A plugin projects to **one capability identity**. With no `artifact:` it loads as a plain
`Capability` (instructions only); with `artifact: <type>` it loads as a `PluginCapability` carrying
the matching builder from `ARTIFACT_BUILDERS` (instructions + the result hook, one id). Genuinely
cross-cutting hooks (`FilterPathGuard`, `ProcessHistory`) are *not* plugins; they stay in `hooks.py`.

## On-disk layout

```
src/orchestrator_agent/capabilities/
  spec.py            # PluginSpec (validates frontmatter, carries instructions) + skills_from_specs
  loader.py          # discover *.md, parse frontmatter -> PluginSpec (body is the prompt, verbatim)
  hooks.py           # build_capabilities() + cross-cutting hooks (FilterPathGuard, history trim)
  behavior/          # shared behaviour: PluginCapability base + artifact mappers + `artifact:` registry
  system_prompt.md   # agent-level system prompt (NOT a plugin) — loaded by load_system_prompt()
  plugins/           # shipped plugins (data, not code) — only plugin files live here
    search.md
    aggregate.md
    entity.md
    export.md
```

`system_prompt.md` is the agent system prompt and sits beside `plugins/`, not in it. Files prefixed `_`
are never loaded as plugins.

## Plugin file format

```markdown
---
id: search
description: Find subscriptions, products, workflows, processes…
advertise: true
a2a_tags: [search, query, fuzzy, semantic]
examples: [Find all active subscriptions]
defer_loading: false          # required: always-on (false) or load on demand (true)
tools: [SEARCH_TOOL]          # tools this plugin OWNS (drives artifact mapping + defer gating)
artifact: query               # shared artifact mapping (query/data/export); omit = instructions-only
---
# Searching
… determine the entity_type, then run the search …
```

## Assembly — verbatim Markdown

`loader.load_plugin_specs()` uses the Markdown body as the prompt directly — no template language,
no tool-name substitution, no includes. How to *use* the tools (filtering, operator choice,
discovery order) lives in the MCP tool descriptions on orchestrator-core, not in the prompt.

**Prompts describe intent and do not name MCP tools.** The model binds "run the search" → the
`search` tool from the tool's own description; the plugin owns exactly the action tool it needs
(unambiguous), and `FilterPathGuard` enforces discover-before-filter. This "fat tools, thin prompts"
model keeps prompts free of tool names with zero substitution machinery (verified live: a search
query still calls `discover_filter_paths`→`search`).

The operator's `AGENT_DOMAIN_CONTEXT` (identifier/field conventions) is appended once to the
agent-level system prompt (`load_system_prompt`), so every capability sees it.

A plugin still *owns* the tools it lists in `tools:` (constants, resolved to live names by
`owned_tool_names` — a typo fails loud at startup). Ownership is **not** about prompt references; it
drives artifact mapping (a tool's results → the declared `artifact:`) and the `DeferredToolGate`
(`hooks.py`), which hides a tool whose owning plugin is deferred and not yet loaded. The full MCP
toolset is passed to the Agent; pydantic-ai's defer hides a capability's *instructions* but not its
tools, so the gate closes that gap — a deferred plugin's tools hide with its instructions, and
`load_capability` reveals both. Tools owned by no plugin are always available.

## The tool-contract chain

The tool names live in `tool_names.py` and are referenced by **code only** — the artifact builders
(`behavior/artifacts.py`) and `FilterPathGuard` key on them; the frontmatter `tools:` declares
ownership by constant. Prompts don't reference them at all. At startup `mcp_client.verify_tool_contract`
lists the live server's tools and raises if any name in `ALL_TOOL_NAMES` is missing — so a rename in
core fails loudly instead of silently breaking artifact mapping or tool ownership.
(An unreachable server is logged as a warning, not raised — that's operational, not drift.)
pydantic-ai has no built-in for this: it only validates tool calls the *model* makes at runtime and
is agnostic about which names our code depends on.

## Extensibility

Plugins load only from the built-in `plugins/` directory. To add one, drop an `<id>.md` file there
and restart — a fork, like any other behaviour change. No env var, no external scan path.

## Non-goals

- **Bespoke behaviour code in a plugin.** Artifact behaviour is *declared* (`artifact: query`) and
  bound to a shared, type-checked builder function in `behavior/` — a plugin reuses a standard
  artifact with no code. A genuinely new artifact type (custom rendering) still means adding an
  `ArtifactType` value and a builder in `ARTIFACT_BUILDERS`; a per-plugin co-located-code escape
  hatch is a future layer, not implemented. Cross-cutting hooks
  (`FilterPathGuard`, history trim, `DeferredToolGate`) stay global in `hooks.py`.
- **Composite/wrapper tools.** A plugin declares ownership of existing MCP tools; it does not (yet)
  expose a new composite tool that runs a multi-step sequence in one call.
- **Hot reload.** Plugins load at startup.
- **A template language / tool-name substitution.** Prompts are verbatim Markdown that describe
  intent; the model binds intent to tools from their descriptions. Composition is data + Python.
