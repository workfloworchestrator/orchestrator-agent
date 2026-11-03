---
render_macros: false
---

# Connect LibreChat

This guide shows how to: Run [LibreChat](https://www.librechat.ai/) with the Orchestrator agent as a chat model,
including workflow forms.

The setup below is the smallest one that works: LibreChat and its database, with the agent as the only model. For
file uploads, search, other models and production settings, see the
[LibreChat documentation](https://www.librechat.ai/docs/local).

## Before you start

- The agent is running and you know its address. See [Getting started](../getting-started.md).
- Docker with Docker Compose is installed.

## Steps

### 1. Create the LibreChat files

Make an empty directory with two files. You do not need to clone LibreChat.

`docker-compose.yml`:

```yaml
services:
  librechat:
    image: registry.librechat.ai/librechat-ai/librechat:v0.8.8
    ports:
      - '3080:3080'
    depends_on:
      - mongodb
    extra_hosts:
      - 'host.docker.internal:host-gateway'
    environment:
      - HOST=0.0.0.0
      - PORT=3080
      - MONGO_URI=mongodb://mongodb:27017/LibreChat
      - DOMAIN_CLIENT=http://localhost:3080
      - DOMAIN_SERVER=http://localhost:3080
      - LIBRECHAT_TEMP_CREDENTIALS_PATH=/app/data/.env.temp
      - ALLOW_REGISTRATION=true
      - SCHEDULES_SINGLE_PROCESS=true
    volumes:
      - ./librechat.yaml:/app/librechat.yaml:ro,z
      - librechat-data:/app/data

  mongodb:
    image: mongo:8.0
    command: mongod --noauth
    volumes:
      - mongodb-data:/data/db

volumes:
  librechat-data:
  mongodb-data:
```
    
`librechat.yaml`:

```yaml
version: 1.3.17

endpoints:
  custom:
    - name: 'WFO agent'
      apiKey: 'unused'
      baseURL: 'http://host.docker.internal:8080/v1'
      models:
        default: [ 'wfo' ]
        fetch: false
      titleConvo: false
      modelDisplayLabel: 'WFO agent'
      headers:
        X-Conversation-Id: '{% raw %}{{LIBRECHAT_BODY_CONVERSATIONID}}{% endraw %}'
        X-Agent-Client: 'librechat'
        # With OpenID sign-in, forward the signed-in user's token (step 3):
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

Use LibreChat 0.8.8 or later: older versions cannot show workflow forms. Keep `X-Agent-Client`, `X-Conversation-Id`,
`askUserQuestion` and `titleConvo` as shown. The
[LibreChat configuration reference](../reference/configuration.md#librechat) explains what each key does. The file
`examples/librechat.yaml` in the agent repository has the same content with comments.

### 2. Point `baseURL` at the agent

LibreChat calls the agent from inside its container, so `localhost` is the container itself, not your machine.

- If the agent listens on port 8080 of your machine, keep `http://host.docker.internal:8080/v1`.
- If the agent runs elsewhere, use an address the container can reach, ending in `/v1`.

### 3. Set up sign-in

How users sign in to LibreChat depends on whether the agent checks tokens. See
[LibreChat authentication](https://www.librechat.ai/v0.8.8/docs/configuration/authentication) for all options.

**The agent runs with `OAUTH2_ACTIVE=false`** (local try-out only). The email login of the compose file above is
enough. Leave the `Authorization` header commented out.

**The agent runs with `OAUTH2_ACTIVE=true`.** LibreChat must sign users in with OpenID Connect against the same
identity provider as the agent and Orchestrator Core, and pass each user's access token on to the agent:

1. Register LibreChat as a client in your identity provider, with the redirect URI
    `http://localhost:3080/oauth/openid/callback` (or your LibreChat address). LibreChat has
    [guides per provider](https://www.librechat.ai/v0.8.8/docs/configuration/authentication/OAuth2-OIDC), such as
    [Keycloak](https://www.librechat.ai/v0.8.8/docs/configuration/authentication/OAuth2-OIDC/keycloak).
2. Add the OpenID settings to `environment` in `docker-compose.yml`, and turn on
    [token reuse](https://www.librechat.ai/v0.8.8/docs/configuration/authentication/OAuth2-OIDC/token-reuse) so
    LibreChat keeps the user's access token:

    ```yaml
          - OPENID_ISSUER=https://idp.example.com/realms/example
          - OPENID_CLIENT_ID=librechat
          - OPENID_CLIENT_SECRET=...
          - OPENID_SESSION_SECRET=...
          - OPENID_CALLBACK_URL=/oauth/openid/callback
          - OPENID_SCOPE=openid profile email offline_access
          - OPENID_REUSE_TOKENS=true
          - ALLOW_SOCIAL_LOGIN=true
          - ALLOW_REGISTRATION=false
          - ALLOW_EMAIL_LOGIN=false
    ```

3. Make sure the access token is one the agent accepts: issued by the identity provider in the agent's
    `OIDC_CONF_URL`, for an audience the agent and Orchestrator Core trust. See
    [Security and observability](security-observability.md).
4. Uncomment the `Authorization` header in `librechat.yaml`.

### 4. Start LibreChat

```shell
docker compose up -d
```

Open <http://localhost:3080> and register an account. After a change to `librechat.yaml` or `docker-compose.yml`, run
`docker compose up -d --force-recreate`.

LibreChat generates its encryption keys and JWT secrets on first start and stores them in the `librechat-data`
volume. Before you run it for real users, set permanent values for `CREDS_KEY`, `CREDS_IV`, `JWT_SECRET` and
`JWT_REFRESH_SECRET`, see [LibreChat credentials](https://www.librechat.ai/docs/configuration/dotenv#credentials-configuration).

## Verify it works

Select the **WFO** model in a new chat and ask *How many subscriptions are there per status?*. You should get a short
answer with a chart. Then ask *Find all active subscriptions* to get a table.

- **The agent cannot be reached.** Check `baseURL` (step 2) and look for the error in `docker compose logs librechat`.
- **The agent answers 401.** The `Authorization` header is missing, or the agent does not accept the token. Check
  step 3.
- **No workflow form opens.** Check the LibreChat version, `askUserQuestion: true` and the `X-Agent-Client` header.

## Next steps

- [Enable embeddings](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#running-it) in
  Orchestrator Core for semantic and hybrid search if you haven't already done so.
- [Teach the agent your domain](domain-context.md) with your identifiers and terminology.
- Learn about [endpoints and protocols](../reference/endpoints.md) to connect your own clients and tools.

## Reference

- [Configuration: LibreChat](../reference/configuration.md#librechat)
- [Endpoints and protocols](../reference/endpoints.md#openai-chat-completions)
- [LibreChat custom endpoint documentation](https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/custom_endpoint)
- [LibreChat authentication](https://www.librechat.ai/v0.8.8/docs/configuration/authentication)
