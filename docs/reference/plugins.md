# Plugins

Each capability of the agent is a plugin: a Markdown file with YAML frontmatter, shipped in the agent under
`capabilities/plugins/`. The body is the prompt. The plugins describe *what* to do and the tool descriptions served by
Orchestrator Core describe *how* to use each tool. See [How it works](../architecture/index.md) for the design.

All plugins are always on. Every plugin is also advertised as a skill in the A2A agent card.

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
| `a2a_tags`, `examples` | Tags and example prompts for the A2A skill |
| `defer_loading` | `false` for always on, `true` to load on demand together with its tools |
| `tools` | The tools the plugin owns |
| `artifact` | Maps the tool results to `query`, `data` or `export` output. Omit for instructions only |

See [Teach the agent your domain](../guides/domain-context.md) for how to change the behaviour of the agent.
