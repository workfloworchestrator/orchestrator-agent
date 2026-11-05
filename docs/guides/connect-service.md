# How to connect another service

This guide shows you how to let the agent answer questions that need data from another system, such as NetBox, a
monitoring system or a CRM.

The Orchestrator Agent itself connects only to the MCP server of Orchestrator Core. Choose a way in based on where
the data belongs:

| If the data | Do this |
| --- | --- |
| Belongs to subscriptions in WFO, for example the contract or the monitoring status of a subscription | Add a route to your orchestrator and [expose it as a tool](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending). You don't need the steps below. |
| Lives in a system of its own | Give the system its own agent and let a planner combine the answers. Follow the steps below. |

## Steps

The example connects NetBox, as the `agents` profile of the
[example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) does.

### 1. Run an MCP server for the service

Find an MCP server for your system, for example in the
[MCP server list](https://github.com/modelcontextprotocol/servers). For NetBox, run
[netbox-mcp](https://github.com/netboxlabs/netbox-mcp-server) over streamable HTTP, with a read-only API token so the
agent cannot change the service:

```yaml
services:
  netbox-mcp:
    image: docker.io/netboxlabs/netbox-mcp-server:latest
    environment:
      NETBOX_URL: http://netbox:8080/
      NETBOX_TOKEN: <read-only NetBox API token>
      TRANSPORT: http
      HOST: 0.0.0.0
      PORT: "8000"
```

The tools are then served at `http://netbox-mcp:8000/mcp`.

### 2. Create an agent with the MCP server as its toolset

Follow [How to create your own agent](create-agent.md), with `http://netbox-mcp:8000/mcp` as its toolset.

### 3. Connect both agents to a planner

Add the new agent and the Orchestrator Agent as peers of a planner, see
[How to connect another agent over A2A](a2a-planner.md).

## Verify it works

Ask the planner a question that needs both systems, for example *Which sites exist in NetBox, and how many nodes are
in WFO?* The answer should combine data from NetBox and from WFO.

## Reference

- [MCP server of Orchestrator Core: extending](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending)
- [Teach the agent your domain](domain-context.md)
