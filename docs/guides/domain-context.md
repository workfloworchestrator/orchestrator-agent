# Teach the agent your domain

!!! info "Goal"
    Give the agent the identifier conventions and terminology of your organisation, so that a question like *Show
    circuit IS4443* is turned into a filter on the right field.

## Domain knowledge

When a user mentions a concrete identifier, such as a customer name, a subscription id or a code like `IS4443`, the
agent uses it as a search key. It looks up which field matches and filters on it with `like`, which tolerates
substrings and typos, and only falls back to ranking when no field clearly matches. The agent cannot infer which
field an identifier belongs to when that is a convention of your organisation. Tell it with `AGENT_DOMAIN_CONTEXT`:

```shell
AGENT_DOMAIN_CONTEXT="Circuit codes look like IS#### and map to the imsCircuitId field — filter with like. Customer references are 8-digit numbers — field customerId."
```

The text is added as a `## Domain Knowledge` section to the system prompt, so every [capability](../architecture/index.md#capabilities) sees it. An empty value
leaves the section out. Restart the agent to apply a change.

Good things to put in it are identifier formats and the field they map to, the meaning of product names and
abbreviations, and terms that people use for things that have another name in the inventory.

??? tip "Let the agent link to your UI and other services"
    Tell the agent where your WFO UI and other systems live, and it puts clickable links in its answers, so you do
    not have to look a subscription or process up yourself:

    ```shell
    AGENT_DOMAIN_CONTEXT="The WFO UI is at http://localhost:3000. Link a subscription as http://localhost:3000/subscriptions/<subscription_id> and a process as http://localhost:3000/workflows/<process_id>. NetBox is at http://localhost:8000."
    ```

## Changing the behaviour further

The behaviour of each [capability](../architecture/index.md#capabilities) is a Markdown prompt in the [plugins](../reference/plugins.md) of the agent. To change
a prompt or add a plugin, fork the agent, edit or add a file in `capabilities/plugins/` and restart. Plugins are not
loaded from outside the agent, and they describe intent rather than tool names. How a tool is used is described by the
tool descriptions in Orchestrator Core. To give the agent data that no tool serves yet, see
[Add your own tools, services and agents](own-tools.md).

## Verify it works

Ask a question with one of your identifiers, for example *Show circuit IS4443*. The result should be filtered on the
field you named. Compare with the same question before you set `AGENT_DOMAIN_CONTEXT`.

## Reference

- [Configuration: domain context](../reference/configuration.md#domain-context)
- [Plugins](../reference/plugins.md)
