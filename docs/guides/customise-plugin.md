# How to customise a plugin

Change the instructions or loading mode of a built-in plugin in your own agent checkout. To supply identifiers and
terminology without maintaining a fork, use [domain context](domain-context.md) instead.

## Before you start

You need a local clone of the Orchestrator Agent repository that you can edit, with the agent configured and running.
Follow [Run the agent with your Orchestrator Core](../getting-started.md) if you have not set it up yet.

## 1. Select the plugin

Choose the plugin for the behavior you want to change from the [plugin reference](../reference/plugins.md).
Its source is `src/orchestrator_agent/capabilities/plugins/<id>.md`, for example `search.md`.

Choose a few requests handled by this plugin and
[capture a baseline](evals.md#2-capture-or-select-a-baseline) before editing, using a dataset that covers them.

If you need to change a Core tool's parameters or usage guidance, edit its route and description in Core instead.
See [Extending the Core MCP server](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending).

## 2. Edit its instructions or metadata

Change the Markdown body to describe the desired behavior. Keep tool usage details in Core's tool descriptions.
For a change to the search plugin's loading mode, retain its required frontmatter and set `defer_loading`:

```yaml
---
id: search
description: Find subscriptions, products, workflows, processes with lenient text/range filters.
a2a_tags: [search, query, fuzzy, semantic]
examples:
  - Find all active subscriptions
defer_loading: true
tools: [SEARCH_TOOL]
artifact: query
---
```

Keep the plugin's existing instructions beneath this frontmatter. Use only symbolic tool constants registered in
`src/orchestrator_agent/tool_names.py`; a live MCP tool name such as `search` is not a replacement for `SEARCH_TOOL`.
Keep the plugin id unique, and consult the [frontmatter reference](../reference/plugins.md#frontmatter) before
changing other fields.

## 3. Validate the change

From the agent checkout, load the plugins and resolve their declared tools:

```shell
uv run --env-file .env python -c \
  "from orchestrator_agent.capabilities import load_plugin_specs; from orchestrator_agent.capabilities.behavior import owned_tool_names; print({p.id: sorted(owned_tool_names(p)) for p in load_plugin_specs()})"
```

Resolve any metadata or constant errors before continuing. Compare the changed agent against your baseline with
[How to benchmark prompt and model changes](evals.md). If the intended requests are missing from the dataset,
[add evaluation cases](add-eval-case.md) and capture a matching baseline for the original configuration first.

## 4. Deploy the edited checkout

Restart a source launch with the updated checkout and `.env` loaded. For a container, build an image from the edited
source and recreate the service with it; restarting an existing image does not include source edits.

## Verify it works

Ask a request affected by the change and inspect the result and tool choices. With `defer_loading: true`, check that
the trace loads the capability before using its owned tools. Confirm that another built-in capability still works.
For tracing setup, see [How to trace the agent](tracing.md).

## Reference

- [Plugins](../reference/plugins.md)
- [Capabilities](../explanation/index.md#capabilities)
