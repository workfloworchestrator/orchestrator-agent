# Enable AI search and embeddings

!!! info "Goal"
    Turn on semantic and hybrid search in Orchestrator Core. Without embeddings the agent works, but it matches on
    fuzzy keywords and structured filters only.

The search index, the retrievers and the embeddings all live in Orchestrator Core. The agent only chooses a retriever
per question: hybrid for identifiers, semantic for descriptive questions, and fuzzy otherwise. When embeddings are
off, it uses fuzzy automatically and stops offering the other two, so no change to the agent is needed.

## Steps

1. Configure an embedding provider in Orchestrator Core:

    ```shell
    EMBEDDING_API_ENABLED=True
    EMBEDDING_API_KEY=sk-...
    EMBEDDING_MODEL=openai/text-embedding-3-small
    EMBEDDING_DIMENSION=1536
    ```

    `EMBEDDING_DIMENSION` must match the model. For a local embedding server, set `EMBEDDING_API_BASE` as well.

2. Build the index. Workflows keep it up to date afterwards, but run this for the first build and after bulk changes:

    ```shell
    python main.py index all
    ```

    `python main.py db upgrade` does not update the index unless you pass `--index`.

3. Restart Orchestrator Core.

## Verify it works

Ask the agent a descriptive question, such as *Which subscriptions are about customer connectivity?*. With embeddings
enabled, the answer is ranked semantically. If you turn them off, the same question still works but is ranked by fuzzy
matching.

## Reference

- [Search](https://workfloworchestrator.org/orchestrator-core/reference-docs/search/)
- [AI / Hybrid Search](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/), including the running without embeddings and the
  local embedding server
- [Teach the agent your domain](domain-context.md) for identifier conventions
