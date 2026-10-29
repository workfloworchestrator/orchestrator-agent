# Teach the agent your domain

!!! info "Goal"
    Give the agent the identifier conventions and terminology of your organisation, so that a question like *Show
    circuit IS4443* is turned into a filter on the right field.

## Steps

When a user mentions a concrete identifier, such as a customer name, a subscription id or a code like `IS4443`, the
agent uses it as a search key. It looks up which field matches and filters on it with `like`, which tolerates
substrings and typos, and only falls back to ranking when no field clearly matches. The agent cannot infer which
field an identifier belongs to when that is a convention of your organisation. Tell it with `AGENT_DOMAIN_CONTEXT`:

```shell
AGENT_DOMAIN_CONTEXT="Circuit codes look like IS#### and map to the imsCircuitId field — filter with like. Customer references are 8-digit numbers — field customerId."
```

The text is added as a `## Domain Knowledge` section to the system prompt, so every capability sees it. An empty value
leaves the section out. Restart the agent to apply a change.

Good things to put in it are identifier formats and the field they map to, the meaning of product names and
abbreviations, and terms that people use for things that have another name in the inventory.

### Retrievers

The agent also chooses how to match for each question:

| Retriever | Used for |
| --- | --- |
| Hybrid (semantic and fuzzy keyword) | Identifiers, codes and names |
| Semantic | Descriptive questions in sentences |
| Fuzzy | Exact tokens, and whenever embeddings are not available |

Semantic and hybrid retrieval need embeddings, see [Enable AI search and embeddings](ai-search.md). Without
them the agent uses fuzzy search and does not offer the other two.

### Changing the behaviour further

The behaviour of each capability is a Markdown prompt in the [plugins](../reference/plugins.md) of the agent. To change
a prompt or add a plugin, fork the agent, edit or add a file in `capabilities/plugins/` and restart. Plugins are not
loaded from outside the agent, and they describe intent rather than tool names. How a tool is used is described by the
tool descriptions in Orchestrator Core. To give the agent data that no tool serves yet, see
[Give the agent your own tools](own-tools.md).

## Verify it works

Ask a question with one of your identifiers, for example *Show circuit IS4443*. The result should be filtered on the
field you named. Compare with the same question before you set `AGENT_DOMAIN_CONTEXT`.

## Reference

- [Configuration: domain context](../reference/configuration.md#domain-context)
- [Plugins](../reference/plugins.md)
