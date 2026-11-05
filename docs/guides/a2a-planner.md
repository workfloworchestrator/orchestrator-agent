# Connect another agent over A2A

To see a planner that combines the Orchestrator Agent with a second agent, follow the
[tutorial](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/).
This guide covers connecting the agent to your own setup. The calling agent is usually a planner, but it can also be
your own agent, see [Create your own agent](create-agent.md).

## Steps

1. Set `BASE_URL` of the Orchestrator Agent to an address that the other agent can reach. The agent card advertises
   this URL, and the caller sends its messages there. Inside Docker or Kubernetes this is the service name, not
   `localhost`.
2. Check that the agent card is served at that address and shows the right URL. The card lists one skill per
   [plugin](../reference/plugins.md):

    ```shell
    curl http://<agent>:8080/.well-known/agent-card.json
    ```

3. Add the Orchestrator Agent as a remote A2A agent of the calling agent, using the base URL of the card.

    === "docker-agent"

        In a [docker-agent](https://github.com/docker/docker-agent) `agent.yaml`, add an `a2a` toolset:

        ```yaml
        toolsets:
          - type: a2a
            name: wfo
            url: http://wfo-agent:8080/
            allow_private_ips: true  # needed for addresses inside a compose network
            timeout: 180
        ```

    === "kagent"

        In [kagent](https://kagent.dev/docs/kagent), add the Orchestrator Agent as an agent tool of the calling agent.

    === "Other runtimes"

        Use the A2A client of your runtime, for example the
        [A2A Python SDK](https://github.com/a2aproject/a2a-python), with the card URL.

    Add the agent of your own service the same way, as a second remote agent next to the Orchestrator Agent, see
    [Connect another service](connect-service.md).

4. Tell the calling agent in its instructions when to use the Orchestrator Agent, for example for questions about
   subscriptions, products, workflows and processes, and when to use your own agent.
5. Restart the calling agent whenever you restart the Orchestrator Agent or change its plugins. Most runtimes read the
   card only at startup.

## Workflow forms

The agent speaks A2A protocol v1.0. Callers on 0.3 are still served, but they cannot fill in workflow forms. Forms
need the [kagent human-in-the-loop extension](../reference/runtime.md#workflow-forms), so that the calling agent can
relay the questions and the approval to the person. The calling agent must request the extension in its call, as
kagent 1.0 and later does. Without it, the calling agent can ask questions but cannot start workflows.

## Reach the planner from your clients

The Orchestrator Agent itself speaks OpenAI chat completions, AG-UI and MCP, so most clients connect to it directly.
A planner built on an A2A runtime often speaks only A2A. A chat UI, a portal, a script or a service that has no A2A
client then cannot reach the planner without something that translates in between.

!!! warning "A translating proxy is also your front door"
    The [example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) solves this with a small
    proxy that turns OpenAI chat completions into A2A. It checks no identity and sets no limits, which is fine for a
    demo on a laptop. In your own stack every request passes this point, so it decides who may call which agent, with
    which credential, and at what cost. Don't copy the demo proxy as is: read
    [Gateways](../architecture/production.md#gateways) first.

## Verify it works

Ask the calling agent a question about WFO, for example *How many active subscriptions are there?*. The answer
should come from the Orchestrator Agent.

## Reference

- [Endpoints and protocols: A2A](../reference/endpoints.md#a2a)
- [Configuration: `BASE_URL`](../reference/configuration.md)
- [Plugins](../reference/plugins.md)
- [Create your own agent](create-agent.md), to build the second agent
