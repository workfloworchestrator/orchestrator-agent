---
render_macros: false
---

# Connect LibreChat

!!! info "Goal"
    Use the agent as a custom endpoint in [LibreChat](https://www.librechat.ai/), including workflow forms.
    You need a running agent and LibreChat 0.8.8 or later, which is the first version with the ask-user tool that
    the workflow forms use.

## Steps

Merge these two entries into your `librechat.yaml`. The file `examples/librechat.yaml` in the agent repository has the
same content with more comments.

```yaml
endpoints:
  custom:
    - name: 'WFO agent'
      apiKey: 'unused'
      baseURL: 'http://orchestrator-agent:8080/v1'
      models:
        default: ['wfo']
        fetch: false
      titleConvo: false
      modelDisplayLabel: 'WFO agent'
      headers:
        X-Conversation-Id: '{% raw %}{{LIBRECHAT_BODY_CONVERSATIONID}}{% endraw %}'
        X-Agent-Client: 'librechat'
        # With OpenID login, forward the signed-in user's token:
        # Authorization: 'Bearer {% raw %}{{LIBRECHAT_OPENID_ACCESS_TOKEN}}{% endraw %}'

modelSpecs:
  list:
    - name: 'wfo'
      label: 'WFO'
      askUserQuestion: true
      preset:
        endpoint: 'WFO agent'
        model: 'wfo'
```

Adjust `baseURL` to the address of the agent as LibreChat can reach it. From inside a container, `localhost` is the
container itself.

- `X-Agent-Client: librechat` is required. The agent opens a workflow form only for a caller that identifies itself as
  LibreChat.
- `X-Conversation-Id` lets the agent keep the memory of a conversation and its open form.
- `askUserQuestion: true` offers the ask-user tool to the agent. Without it no workflow form is opened.
- `titleConvo: false` stops LibreChat from asking the agent for a chat title before the first answer.
- When the agent runs with `OAUTH2_ACTIVE=true`, forward the token of the signed-in user with the `Authorization`
  header. The agent passes it on to Orchestrator Core.

Restart LibreChat after you change `librechat.yaml`.

## Verify it works

Select the **WFO** model in a new chat and ask *How many subscriptions are there per status?*. You should get a short
answer with a chart. Then ask *Find all active subscriptions* to get a table.

If the agent cannot be reached, check that the agent is reachable from the LibreChat container under the host name
in `baseURL`. When LibreChat runs in a container on the same host as the agent, use an address of the host that the
container can resolve rather than `localhost`.

## Reference

- [Configuration: LibreChat](../reference/configuration.md#librechat)
- [Endpoints and protocols](../reference/endpoints.md#openai-chat-completions)
- [LibreChat custom endpoint documentation](https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/custom_endpoint)
