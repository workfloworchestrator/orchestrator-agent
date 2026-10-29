# Add your own tools, services and agents

The Orchestrator Agent answers from the tools of Orchestrator Core. When a question needs more than those tools
offer, you can extend it in three steps, from small to large:

1. **Add a tool.** Expose a route of your orchestrator, and the agent can use it at once.
2. **Connect another service.** Bring in the data of a system outside WFO, such as NetBox or a CRM.
3. **Create your own agent.** Run an agent with its own tools, instructions and model next to the Orchestrator Agent.

The last two steps lead to an agentic environment of your own: several agents, each responsible for one system, with
a planner that combines their answers over [A2A](https://a2a-protocol.org/latest/). This page gives the pointers to set
that up. The details are in the linked guides and in the documentation of each runtime.

Pick the section that matches where the data or the behaviour lives:

| You want to | Section |
| --- | --- |
| Use data that only your orchestrator has | [Expose a route of your orchestrator](#expose-a-route-of-your-orchestrator) |
| Use data of another system, such as NetBox, a monitoring system or a CRM | [Connect other services](#connect-other-services) |
| Add an agent with its own tools, instructions or model | [Create your own agent](#create-your-own-agent) |

## Expose a route of your orchestrator

The agent uses the tools that the MCP server of Orchestrator Core serves. A tool is a FastAPI route with the tag
`agent-exposed`. Orchestrator Core turns it into an MCP tool, and the agent offers it to the model without any change
in the agent.

1. Add a route to your orchestrator and tag it:

    ```python
    from fastapi import APIRouter

    from orchestrator.core.mcp.server import AGENT_EXPOSED_TAG, READONLY_TOOL

    router = APIRouter()


    @router.get(
        "/customers/{customer_id}/contracts",
        tags=[AGENT_EXPOSED_TAG],
        operation_id="list_customer_contracts",
        openapi_extra=READONLY_TOOL,
    )
    def list_customer_contracts(customer_id: str) -> list[Contract]:
        """List the contracts of one customer, newest first.

        Use this when the user asks which contracts or services a customer has.
        """
        ...
    ```

    - `operation_id` becomes the name of the tool.
    - The docstring becomes the description, and the model picks the tool by it. Say when to use the tool, not only
      what it returns. See [How it works](../architecture/index.md#orchestrator-core-search-index-and-mcp-server) for
      why the tool description carries this.
    - The Pydantic models of the route become the parameter schema.
    - `openapi_extra` sets the hints that clients use to judge safety: `READONLY_TOOL`, `WRITE_TOOL`,
      `IDEMPOTENT_WRITE_TOOL` or `DESTRUCTIVE_TOOL`, all from `orchestrator.core.mcp.server`.

2. Restart the orchestrator. Orchestrator Core builds the list of tools once, when `OrchestratorCore(...)` is created.

!!! warning "Expose read-only tools only"
    The agent passes a new tool to the model as it is, in every conversation. It only hides the write tools of
    Orchestrator Core that it knows by name. A tool of yours that changes something would be called by the model
    without the approval of a person. See [why workflows need approval](../architecture/index.md#why-workflows-need-approval).

### Verify it works

Open the API docs of your orchestrator at `http://<core>/api/docs#/agent-exposed`. Every tool is listed there with
its name (the `operationId`), its description and its parameters. Your route should be among them. Then ask the
agent a question that needs the tool, for example *Which contracts does customer 12345678 have?*

## Connect other services

The Orchestrator Agent connects to one MCP server: the one of Orchestrator Core, set with `WFO_CORE_MCP_URL`. It does
not load the MCP servers of other systems. There are two ways to bring another service in:

- **Through your orchestrator.** When the data belongs to the subscriptions in WFO, for example the contract or the
  monitoring status of a subscription, add a route to your orchestrator that fetches it and
  [expose the route](#expose-a-route-of-your-orchestrator). The agent then answers from one place.
- **Through an agent of its own.** When the service is a system of its own, give it its own agent and let a planner
  combine the answers. This keeps each agent small, with its own tools and instructions.

For the second way:

1. Find or build an MCP server for the service. Many systems have one, for example
   [netbox-mcp](https://github.com/netboxlabs/netbox-mcp-server) for NetBox. See the
   [MCP server list](https://github.com/modelcontextprotocol/servers).
2. Create an agent with that MCP server as its toolset, see [Create your own agent](#create-your-own-agent).
3. Add it as a peer of a planner, next to the Orchestrator Agent, see
   [Use the agent from an A2A planner](a2a-planner.md).

The agents profile of the [example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) does
exactly this for NetBox: `inventory-agent` reads NetBox through `netbox-mcp`, and `planner-agent` combines its answers
with those of the Orchestrator Agent.

!!! warning "The guards of the Orchestrator Agent do not apply to other agents"
    The approval of workflows and the hiding of write tools are part of the Orchestrator Agent. An agent of your own
    calls whatever tools its MCP server offers. Give it read-only tools, or let its runtime ask a person before each
    call that changes something.

## Create your own agent

An agent that works with the Orchestrator Agent needs three things: a runtime, tools over MCP, and an A2A endpoint
with an agent card. Choose the runtime that fits your environment:

| Runtime | Fits | Docs |
| --- | --- | --- |
| [docker-agent](https://github.com/docker/docker-agent) | Compose or a VM. An agent is an `agent.yaml` with its model, instructions and toolsets | [docker-agent](https://github.com/docker/docker-agent) |
| [kagent](https://kagent.dev/) | Kubernetes. Agents and MCP servers are declared as resources | [kagent docs](https://kagent.dev/docs/kagent) |
| Your own code | Full control over the agent loop, for example with pydantic-ai or the A2A Python SDK | [pydantic-ai A2A](https://ai.pydantic.dev/a2a/), [A2A Python SDK](https://github.com/a2aproject/a2a-python) |

Then, for any runtime:

1. **Give it tools over MCP.** Point the agent at the MCP servers of the systems it works with. To reach WFO from your
   own agent, prefer delegating to the Orchestrator Agent over A2A. Connecting to the MCP server of Orchestrator Core
   directly bypasses the guards above, see [Read tools and write tools](../architecture/index.md#read-tools-and-write-tools).
2. **Write its instructions.** Describe what the agent is for and when it should use which tool. Keep how a tool is
   used in the description of the tool.
3. **Serve it over A2A.** The runtime serves an agent card at `/.well-known/agent-card.json`. The card lists the
   skills of the agent, and a planner reads them to decide when to call it. Advertise an address that the other
   agents can reach, not `localhost` inside a container.
4. **Pass the identity of the user on.** When your services require authentication, forward the token of the user
   instead of a service account, so the agent can only do what the user may do. See
   [Security and observability](security-observability.md).
5. **Connect it to a planner.** Add the agent card URL as a peer of the planner and tell the planner when to use it,
   see [Use the agent from an A2A planner](a2a-planner.md).

To give people a chat frontend on top, put the planner behind an OpenAI-compatible endpoint, because LibreChat has no
A2A client. The example orchestrator uses `a2a-proxy` for this, see the
[tutorial](https://workfloworchestrator.org/learn/orchestrator-agent/tutorial/).

### Verify it works

1. Fetch the agent card of your agent with `curl http://<agent>/.well-known/agent-card.json`. Check the URL and the
   skills in it.
2. Ask the planner a question that needs your agent and the Orchestrator Agent together. The answer should combine
   both.
3. Before you change prompts or models, add the questions to a benchmark, see
   [Benchmark prompt and model changes](evals.md).

## Reference

- [MCP server of Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/), with the default tool set
- [Endpoints and protocols: A2A](../reference/endpoints.md#a2a)
- [Teach the agent your domain](domain-context.md)
- [Running the agent in your own stack](../architecture/production.md), for deployment shapes and identity
