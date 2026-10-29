# Getting started

Run the agent next to your own Orchestrator Core and ask it a first question. To try it on a ready-made setup
instead, follow the [tutorial on the example orchestrator](https://workfloworchestrator.org/learn/orchestrator-agent/tutorial/).

## Prerequisites

- An Orchestrator Core with its MCP server enabled (`MCP_ENABLED=True`, installed with the `mcp` extra). See the
  [MCP server reference](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/).
- Network access to the Orchestrator Core database, because the agent connects to it with `DATABASE_URI`.
- An API key for an LLM, or an OpenAI-compatible endpoint of your own. See [Use your own LLM](guides/own-llm.md).
- (optional) AI search and embeddings enabled in Orchestrator Core, see
  [Running AI search](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#running-it).

!!! tip "Why enable embeddings"
    The search index and the embeddings live in Orchestrator Core, so you set them up there and not in the agent.
    The agent works without them: it then matches on fuzzy keywords and structured filters, and needs no other
    configuration.

    With embeddings, the agent can also search on meaning. A question such as *Which subscriptions are about
    customer connectivity?* finds subscriptions that don't contain those words, and a search for `amsterdam router`
    finds a subscription described as "AMS core node". The agent then picks a retriever per question: hybrid for
    identifiers and names, semantic for descriptive questions, and fuzzy otherwise. See
    [Teach the agent your domain](guides/domain-context.md#retrievers).

## 1. Run the agent

The agent is published as a container image on the
[GitHub Container Registry](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent).
You can also run it from source with [uv](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/workfloworchestrator/orchestrator-agent.git
cd orchestrator-agent
cp .env.example .env
uv sync
uv run uvicorn orchestrator_agent.app:app --port 8080
```

Or build and run the container with `docker compose up --build`.

## 2. Point it at Orchestrator Core

Edit `.env`. These are the settings you need for a first run:

```shell
DATABASE_URI=postgresql+psycopg://user:pass@localhost/orchestrator
WFO_CORE_MCP_URL=http://localhost:8080/mcp
AGENT_MODEL=openai:gpt-4o
AGENT_API_KEY=<your key>
OAUTH2_ACTIVE=false
```

Both the agent and Orchestrator Core default to port 8080, so run one of them on another port or on another host and
adjust `WFO_CORE_MCP_URL` and `BASE_URL` to match. Keep `OAUTH2_ACTIVE=false` for a local try-out only, see
[Security and observability](guides/security-observability.md) before you expose the agent. All settings are listed in
[Configuration](reference/configuration.md).

At startup the agent checks the tools of the Orchestrator Core MCP server. If a tool it needs is missing, it stops
with an error that names the tool.

## 3. Ask a first question

Check that the agent is up:

```shell
curl http://localhost:8080/health
```

Then ask a question with the MCP demo client from the repository:

```shell
uv run demos/mcp_client.py search "active subscriptions"
```

Or send a chat completion to the OpenAI-compatible endpoint:

```shell
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "wfo", "messages": [{"role": "user", "content": "How many subscriptions are there per status?"}]}'
```

## Next steps

- [Connect LibreChat](guides/librechat.md) to chat with the agent.
- [Enable embeddings](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#running-it) in
  Orchestrator Core for semantic and hybrid search if you haven't already done so.
- [Teach the agent your domain](guides/domain-context.md) with your identifiers and terminology.
- [Endpoints and protocols](reference/endpoints.md) for the other ways to reach the agent.
