# How to connect another agent over A2A

This guide shows you how to let another agent, usually a planner, call the Orchestrator Agent over
[A2A](https://a2a-protocol.org/latest/). To see a working planner first, follow the
[tutorial](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/). To build the
second agent, see [How to create your own agent](create-agent.md).

## Steps

1. Set `BASE_URL` of the Orchestrator Agent to an address that the calling agent can reach. Inside Docker or
   Kubernetes, use the service name, not `localhost`.
2. Check that the agent card shows that URL:

    ```shell
    curl http://<agent>:8080/.well-known/agent-card.json
    ```

3. Add the Orchestrator Agent as a remote A2A agent of the calling agent, using the base URL of the card:

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

    Add the agent of your own service the same way, next to the Orchestrator Agent.

4. In the instructions of the calling agent, say when to use the Orchestrator Agent, for example for questions about
   subscriptions, products, workflows and processes, and when to use your other agents.
5. If the Orchestrator Agent authenticates requests, have the calling agent send the user's token twice: in the
   `Authorization` header to pass the agent's check, and as `auth_token` in the message metadata so that Orchestrator
   Core sees the user. Without the metadata, calls to Orchestrator Core run as the agent's service account. See
   [Authentication and authorization](../architecture/auth.md#outbound-who-the-agent-acts-as).
6. If users should start workflows through the calling agent, make it request the
   [kagent human-in-the-loop extension](../reference/runtime.md#workflow-forms), as kagent 1.0 and later does. Without
   it, the calling agent can ask questions but cannot start workflows.
7. If your clients do not speak A2A, such as most chat UIs, put a gateway in front of the planner. Do not reuse the
   proxy of the example orchestrator: it checks no identity and sets no limits. See
   [Gateways](https://workfloworchestrator.org/learn/orchestrator-agent/production/#gateways).

Restart the calling agent whenever you restart the Orchestrator Agent or change its plugins: most runtimes read the
card only at startup.

## Verify it works

1. Ask the calling agent a question about WFO, for example *How many active subscriptions are there?*
2. Check in the log or trace of the calling agent that it called the `wfo` peer, and that the answer contains the
   count it returned.
3. If you set up workflow forms, ask it to start a workflow. It should relay the form questions and ask for approval
   before the workflow starts.

## Reference

- [Endpoints and protocols: A2A](../reference/endpoints.md#a2a)
- [Configuration: `BASE_URL`](../reference/configuration.md)
- [Plugins](../reference/plugins.md)
