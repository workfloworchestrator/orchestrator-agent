# How to secure the agent

This guide shows you how to let only signed-in users call the agent, and have it call Orchestrator Core as those
users. To learn how the checks work, see [Authentication and authorization](../architecture/auth.md).

## Steps

### 1. Authenticate incoming requests

Set the details of your identity provider:

```shell
OAUTH2_ACTIVE=true
OIDC_BASE_URL=https://idp.example.com
OIDC_CONF_URL=https://idp.example.com/.well-known/openid-configuration
OAUTH2_RESOURCE_SERVER_ID=<client id>
OAUTH2_RESOURCE_SERVER_SECRET=<client secret>
```

### 2. Authorize requests

=== "With OPA"

    Set the decision endpoint of your policy:

    ```shell
    OPA_URL=https://opa.example.com/v1/data/agent/allow
    ```

=== "Without OPA"

    Turn authorization off:

    ```shell
    OAUTH2_AUTHORIZATION_ACTIVE=false
    ```

### 3. Authenticate towards Orchestrator Core

=== "Orchestrator Core requires a token"

    Set the token endpoint for the agent's service account:

    ```shell
    OAUTH2_TOKEN_URL=https://idp.example.com/oauth2/token
    ```

    In Orchestrator Core, give that client only the rights it needs, ideally read-only.

=== "Orchestrator Core accepts calls without a token"

    ```shell
    OAUTH2_OUTBOUND_ACTIVE=false
    ```

### 4. Send the user's token from your client

Send the user's token in the `Authorization` header. If you connect over A2A, also put it in the message metadata
as `auth_token`.

For a client-specific setup, see [Connect LibreChat](librechat.md), [Use the agent with MCP](mcp-clients.md) or
[Build a frontend with AG-UI](ag-ui.md).

### 5. Restrict where the agent listens

1. Bind the published ports to a specific address.
2. Put the agent behind a reverse proxy that terminates TLS.
3. Keep the `DATABASE_URI` credentials in a secret store.
4. Give the agent a read-only database role, if your deployment allows it.

## Verify it works

- Call an endpoint without a token. The agent should answer 401. `GET /health` stays open for monitoring.
- Call it with a valid token. A 403 with a decision id means OPA denied the request; a 503 means OPA could not be
  reached or `OPA_URL` is not set.
- Ask a question that starts a workflow and check in Orchestrator Core that the workflow was started by your user,
  not by the service account.

## Reference

- [Configuration: authentication](../reference/configuration.md#authentication)
- [Authentication and authorization](../architecture/auth.md)
