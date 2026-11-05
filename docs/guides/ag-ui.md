# Build a frontend with AG-UI

## Steps

1. Point your AG-UI client at `POST /agui` of the agent. The response is a stream of server-sent events.
2. Send a thread id with each request. A follow-up on the same thread keeps its history, so *export them* can refer
   to the results of the previous question.
3. When authentication is on, send the bearer token in the `Authorization` header.
4. Render the events. Besides text, the stream carries the results of searches and aggregations as structured data:
   a query result for a table or chart, a data artifact for a single entity and an export artifact with a download
   URL. The agent also puts a rendered table or chart in the text of the answer, so a client that only shows text
   still shows the result.

The AG-UI endpoint runs the agent without the workflow form-fill skill, so it has no approved path to start workflows
and its model can call the write tools of Orchestrator Core directly. To start workflows from chat, see
[Start workflows from chat](workflows-from-chat.md).

Tools supplied by the frontend in `RunAgentInput.tools` are not available to the model. Its tools come from
Orchestrator Core.

## Verify it works

Use the demo client of the agent repository:

```shell
# stream a search query
uv run demos/agui_client.py "find active subscriptions"

# follow-up on the same thread
uv run demos/agui_client.py "export them" <thread-id>
```

## Reference

- [Endpoints and protocols](../reference/endpoints.md#ag-ui)
- [AG-UI in pydantic-ai](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/), for how events are mapped
