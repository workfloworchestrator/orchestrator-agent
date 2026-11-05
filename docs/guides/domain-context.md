# How to teach the agent your domain

This guide shows you how to tell the agent about your organisation's identifiers and terms, so that a question like
*Show circuit IS4443* becomes a filter on the right field. To learn why the agent needs this, see
[How a question becomes a search](../architecture/index.md#how-a-question-becomes-a-search).

## Steps

### 1. Describe your conventions

Write a few sentences about the identifiers and terms that people in your organisation use:

- identifier formats and the field each one maps to
- what your product names and abbreviations mean
- terms people use for things that have a different name in the inventory

If you want the agent to link to your WFO UI and other systems in its answers, also say where they live and how to
build a link to a subscription or a process.

### 2. Set `AGENT_DOMAIN_CONTEXT`

```shell
AGENT_DOMAIN_CONTEXT="Circuit codes look like IS#### and map to the imsCircuitId field — filter with like. Customer references are 8-digit numbers — field customerId."
```

With links to your UI:

```shell
AGENT_DOMAIN_CONTEXT="The WFO UI is at http://localhost:3000. Link a subscription as http://localhost:3000/subscriptions/<subscription_id> and a process as http://localhost:3000/workflows/<process_id>. NetBox is at http://localhost:8000."
```

### 3. Restart the agent

The agent reads the value at startup, so restart it after every change.

## Verify it works

1. Ask a question that contains one of your identifiers, for example *Show circuit IS4443*.
2. Check that the result is filtered on the field you named. In a trace, the search tool call shows the filter.
3. If you added links, check that the answer links to your UI.

## Next steps

- To change how a capability works, not just what it knows, edit the plugin prompts, see
  [Customise a plugin](../reference/plugins.md#customise-a-plugin).
- To give the agent data that no tool serves yet,
  [expose a route of your orchestrator as a tool](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending)
  or [connect another service](connect-service.md).

## Reference

- [Configuration: domain context](../reference/configuration.md#domain-context)
- [Plugins](../reference/plugins.md)
