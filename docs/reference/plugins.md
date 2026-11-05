# Plugins

Each plugin is loaded as one [capability](../architecture/index.md#capabilities) of the agent: a Markdown file with YAML frontmatter, shipped in the agent under
`capabilities/plugins/`. The body is the prompt. The plugins describe *what* to do and the tool descriptions served by
Orchestrator Core describe *how* to use each tool. See [How the agent fits on top of Orchestrator Core](../architecture/index.md) for the design.

All built-in plugins are always on. The workflow plugin is present only where the form-fill skill runs: A2A and `/v1`.
Every plugin is also advertised as a skill in the A2A agent card, unless it sets `advertise: false`.

| Plugin | Tools from Orchestrator Core | Output |
| --- | --- | --- |
| [search](#search) | `search` | Table of results |
| [aggregate](#aggregate) | `aggregate` | Chart, or a number |
| [entity](#entity) | `resolve_entity` | Details of one entity |
| [export](#export) | `export_query` | Download link |
| [workflow](#workflow) | `list_workflows`, `start_workflow_form` | A form to fill in, then a started process |

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

Prepares a download of the results of an earlier query, by default the most recent one, and returns a download URL.
The agent only exports when the user explicitly asks for it, with words such as export, download or CSV.

Example: *Export the last search results as CSV.*

## workflow

Starts a workflow, whether it creates, modifies or terminates something, or runs a task. The agent lists the
workflows, picks the right one and hands over to the form-fill skill, which asks the user the questions of the form
and asks for approval before it starts the process. The agent never starts a workflow on chat text alone. See
[Start workflows from chat](../guides/workflows-from-chat.md).

Example: *Start the terminate workflow for subscription abc123.*

## Frontmatter

| Key | Description |
| --- | --- |
| `id` | Name of the plugin |
| `description` | Shown in the A2A agent card |
| `advertise` | `true` (the default) to advertise the plugin as a skill in the A2A agent card |
| `a2a_tags`, `examples` | Tags and example prompts for the A2A skill |
| `defer_loading` | `false` for always on, `true` to load on demand: the instructions and the tools stay hidden until the model loads the [capability](../architecture/index.md#capabilities) |
| `tools` | The tools the plugin owns, for result mapping and deferred loading |
| `artifact` | Maps the tool results to `query`, `data` or `export` output. Omit for instructions only |

## Customise a plugin

The agent only loads plugins from its own `capabilities/plugins/` directory. You cannot add one from outside, so to
change a prompt or add a plugin, fork the agent, edit or add a file in that directory, and restart.

A plugin says *what* the agent should do, not *which tool* to call. How each tool is used comes from the tool
descriptions that Orchestrator Core serves. To change how a tool is used, change its description in Orchestrator
Core rather than naming the tool in a plugin.

To give the agent knowledge about your identifiers and terminology without forking, see
[Teach the agent your domain](../guides/domain-context.md).
