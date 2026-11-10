# Plugins

Each plugin is loaded as one [capability](../architecture/index.md#capabilities) of the agent: a Markdown file with YAML frontmatter, shipped in the agent under
`capabilities/plugins/`. The body is the prompt. The plugins describe *what* to do and the tool descriptions served by
Orchestrator Core describe *how* to use each tool. See [How the agent fits on top of Orchestrator Core](../architecture/index.md) for the design.

All built-in plugins are always on. The workflow plugin is present where the form-fill skill runs: A2A, `/v1` and AG-UI.
Every plugin is also advertised as a skill in the A2A agent card, unless it sets `advertise: false`.

| Plugin | Tools it uses | Output |
| --- | --- | --- |
| [search](#search) | `search` | Table of results |
| [aggregate](#aggregate) | `aggregate` | Chart, or a number |
| [entity](#entity) | `resolve_entity` | Details of one entity |
| [export](#export) | `export_query` | Prepared export; downloadable through the AG-UI chats API |
| [workflow](#workflow) | `list_workflows`, `start_workflow_form` | A form to fill in, then a started process |

`start_workflow_form` is a local handoff tool supplied by the form-fill capability. The other tools in this table
come from Orchestrator Core. The skill itself calls Core's form and workflow-start tools after the handoff.

The agent also uses `discover_filter_paths` and `get_valid_operators` before it filters or groups. These belong to no
plugin and are always available.

## search

Finds subscriptions, products, workflows and processes. The agent determines the entity type, passes the user's text
and any filters the question implies, and summarises the outcome in one sentence. The table of results is rendered
by the agent itself, so the answer does not repeat the rows.

Example: *Find all active subscriptions.*

## aggregate

Counts, or computes sum, average, minimum and maximum, optionally grouped (`group_by`) or bucketed in time
(`temporal_group_by`). A grouped result is rendered as a chart, a bare count is stated as a number.

Example: *How many subscriptions per status?*

## entity

Fetches a single entity by id or id prefix. A unique match returns the entity. Multiple matches return a list of
candidates and the agent asks the user to pick one. Viewing an entity is never treated as an export.

Example: *Show details for subscription abc123.*

## export

Prepares a download of the results of an earlier query, by default the most recent one. The plugin instructions
limit this to an explicit request, with words such as export, download or CSV; this is prompt guidance, not a
separate authorization check.

The download URL is retained in `ExportArtifact` metadata and delivered to AG-UI clients through the
[chats export endpoint](endpoints.md#chats-api). The plugin removes Core's standard `download_path` and `message`
fields from the model-visible result and tells the model to confirm that the download was prepared without
writing a link. Chat completions do not return that artifact metadata, and A2A/MCP send the model-visible payload,
so those interfaces currently do not deliver the standard export URL. Use AG-UI for the built-in download flow.

Example: *Export the last search results as CSV.*

## workflow

Hands off a workflow start, whether it creates, modifies or terminates something, or runs a task. The agent lists the
workflows, picks the right one and hands over to the form-fill skill, which asks the user the questions of the form
and asks for approval before it starts the process. The agent never starts a workflow on chat text alone. See
[Start workflows from chat](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/workflows-from-chat/).

Example: *Start the terminate workflow for subscription abc123.*

## Frontmatter

| Key | Required / default | Description |
| --- | --- | --- |
| `id` | Required | Stable plugin name |
| `description` | Required | Capability description and A2A skill description |
| `advertise` | `true` | Advertise an A2A skill and, where eligible, an MCP entry-point tool |
| `a2a_tags` | `[]` | A2A tags; the advertised skill falls back to the plugin id when empty |
| `examples` | `[]` | Example prompts for the A2A skill and MCP tool description |
| `defer_loading` | Required | `false` for always on, `true` to hide instructions and owned tools until the model loads the [capability](../architecture/index.md#capabilities) |
| `tools` | `[]` | Symbolic constants from `tool_names.py`, such as `SEARCH_TOOL`, resolved to Core tool names for result mapping and deferred loading |
| `artifact` | Omitted | Maps tool results to `query`, `data` or `export` output; omitted means instructions only |

Unknown frontmatter keys are rejected when specs are loaded; unknown tool constants fail when their ownership is
resolved during capability construction. The Markdown body supplies the
instructions; it is not a template. Files whose names start with `_` are not loaded.

## Customise a plugin

The application loads plugins from its packaged `capabilities/plugins/` directory. There is no environment setting
for an external plugin directory. The body describes intent; Core's tool descriptions describe how the tools are used.

For changes, see [How to customise a plugin](../guides/customise-plugin.md). Deployment-specific identifiers and
terminology can be supplied without a fork through [domain context](../guides/domain-context.md).
