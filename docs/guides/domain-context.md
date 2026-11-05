# Teach the agent your domain

Tell the agent which field your own identifiers belong to, so that a question like *Show circuit IS4443* becomes a
filter on the right field.

The agent treats an identifier in a question as a search key. It looks for the field that matches and filters on it
with `like`, which allows for substrings and typos. If no field clearly matches, it ranks results instead. It cannot
work out conventions that only your organisation knows, such as which field holds a circuit code. You give it that
knowledge with `AGENT_DOMAIN_CONTEXT`.

## 1. Describe your conventions

Write a few sentences about the identifiers and terms that people in your organisation use:

- identifier formats and the field each one maps to
- what your product names and abbreviations mean
- terms people use for things that have a different name in the inventory

## 2. Set `AGENT_DOMAIN_CONTEXT`

```shell
AGENT_DOMAIN_CONTEXT="Circuit codes look like IS#### and map to the imsCircuitId field — filter with like. Customer references are 8-digit numbers — field customerId."
```

The text is added to the system prompt as a `## Domain Knowledge` section, so every
[capability](../architecture/index.md#capabilities) sees it. If the value is empty, the section is left out.

??? tip "Let the agent link to your UI and other services"
    Tell the agent where your WFO UI and other systems live, and it puts clickable links in its answers, so you do
    not have to look a subscription or process up yourself:

    ```shell
    AGENT_DOMAIN_CONTEXT="The WFO UI is at http://localhost:3000. Link a subscription as http://localhost:3000/subscriptions/<subscription_id> and a process as http://localhost:3000/workflows/<process_id>. NetBox is at http://localhost:8000."
    ```

## 3. Restart the agent

The agent reads the value at startup, so restart it after every change.

## 4. Verify it works

Ask a question that contains one of your identifiers, for example *Show circuit IS4443*. The result should be
filtered on the field you named. Compare it with the result of the same question without `AGENT_DOMAIN_CONTEXT`.

## Going further

`AGENT_DOMAIN_CONTEXT` adds knowledge but does not change how a capability works. To change that, edit the plugin
prompts, see [Customise a plugin](../reference/plugins.md#customise-a-plugin). To give the agent data that no tool
serves yet, [expose a route of your orchestrator as a tool](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending)
or [connect another service](connect-service.md).

## Reference

- [Configuration: domain context](../reference/configuration.md#domain-context)
- [Plugins](../reference/plugins.md)
