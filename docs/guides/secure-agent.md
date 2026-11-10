# How to configure authentication and authorization

Configure who may call the agent and which identity it uses when calling Orchestrator Core. Have a running agent
and the client details from your identity provider ready. Edit the agent's `.env` for a source launch, or the
environment supplied to your container. See [Authentication and authorization](../architecture/auth.md) for the design.

## Steps

### 1. Authenticate incoming requests

Set the details of your identity provider:

```shell
OAUTH2_ACTIVE=true
OIDC_BASE_URL=https://idp.example.com
OIDC_CONF_URL=https://idp.example.com/.well-known/openid-configuration
```

### 2. Authorize requests

=== "With OPA"

    Set the decision endpoint of your policy:

    ```shell
    OAUTH2_AUTHORIZATION_ACTIVE=true
    OPA_URL=https://opa.example.com/v1/data/agent/allow
    ```

=== "Without OPA"

    Turn authorization off:

    ```shell
    OAUTH2_AUTHORIZATION_ACTIVE=false
    ```

### 3. Authenticate towards Orchestrator Core

=== "Orchestrator Core requires a token"

    Enable the fallback service account and set its token endpoint. Startup's tool-contract check uses this account
    before there is a user request:

    ```shell
    OAUTH2_OUTBOUND_ACTIVE=true
    OAUTH2_RESOURCE_SERVER_ID="<client id>"
    OAUTH2_RESOURCE_SERVER_SECRET="<client secret>"
    OAUTH2_TOKEN_URL=https://idp.example.com/oauth2/token
    ```

    Give that client only the Core permissions needed by requests that use it. If it only handles questions, use
    read permissions. Approved workflow starts need the appropriate workflow permission on the identity used for
    that call. Do not grant broad write access merely to get a health or search check to pass.

=== "Orchestrator Core accepts calls without a token"

    ```shell
    OAUTH2_OUTBOUND_ACTIVE=false
    ```

### 4. Send the user's token from your client

For chat completions and AG-UI, send the user's token in the `Authorization` header. For A2A, send it both in that
header and in the message metadata as `auth_token`. Configure Core to accept and authorize the forwarded token.

The agent's MCP endpoint authenticates the caller but does not forward their token to Core. Use a service account
whose permissions are appropriate for every caller allowed to use that endpoint.

For client details, see [Connect LibreChat](librechat.md), [Use the agent with MCP](mcp-clients.md) or the
[AG-UI endpoint reference](../reference/endpoints.md#ag-ui).

### 5. Restrict where the agent listens

1. Bind the published ports to a specific address.
2. Put the agent behind a reverse proxy that terminates TLS.
3. Keep the `DATABASE_URI` credentials in a secret store.
4. Give the database role the access needed to read and insert agent run records and state snapshots. The AG-UI
   chat-deletion endpoint also needs `DELETE` on `graph_snapshots` and `agent_runs`. A wholly read-only role cannot
   run the agent. Core's MCP identity and this database role serve different purposes.
5. Restrict direct access to the agent to trusted clients. Have A2A and chat-completions clients generate and isolate
   their conversation identifiers; those adapters do not check user ownership. AG-UI and its chats API check the
   authenticated subject against the stored owner. With authentication off, its threads are unowned and shared.
6. Configure `WFO_CORE_API_URL` only when form options need a different base URL from Core's MCP origin. Add a host
   to `WFO_OPTIONS_HOSTS` only if it may receive the user's or service account's bearer token. See
   [Configuration](../reference/configuration.md#connection).

Restart the source process with `uv run --env-file .env uvicorn orchestrator_agent.app:app --host 127.0.0.1 --port 8080`,
or recreate your container so it receives the updated environment. For limits and other deployment controls, see
[Running the agent in your own stack](https://workfloworchestrator.org/learn/orchestrator-agent/production/).

## Verify it works

- Call a protected endpoint without a token, such as `GET /v1/models`. It should answer 401. `GET /health` and the
  A2A agent card stay open for monitoring and discovery.
- Call it with a valid token. A 403 with a decision id means OPA denied the request. A 503 can mean OPA could not be
  reached, `OPA_URL` is unset, or OIDC discovery returned a non-200 response; check the service logs to distinguish them.
- Use a user-token request to query Core and confirm its recorded identity. Through a configured LibreChat or A2A
  client, complete and approve a workflow form and confirm the start was attributed to that user.
- For an MCP request, confirm Core sees the intended service account.

## Reference

- [Configuration: authentication](../reference/configuration.md#authentication)
- [Authentication and authorization](../architecture/auth.md)
