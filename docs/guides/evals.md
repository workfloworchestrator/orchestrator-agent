# Benchmark prompt and model changes

!!! info "Goal"
    Check that a change to a prompt, a model or a plugin does not make the agent pick the wrong tools.

## Steps

1. Start the backing services, an Orchestrator Core with the MCP server and a seeded database, and install the
   dependencies. From the `evals/` folder of the agent repository:

    ```shell
    docker compose -f docker-compose.yml up -d --wait
    uv sync
    cp .env.example .env
    ```

    Set `AGENT_API_KEY` in `.env` and pin `AGENT_MODEL`. Scores are only comparable on the same model.

2. Run the golden set, or a single case as a smoke test:

    ```shell
    uv run python run_eval.py
    uv run python run_eval.py --cases fuzzy-search-nodes
    ```

3. Compare with the committed baseline. A case that drops more than the tolerance, or an assertion that starts to
   fail, is reported by name:

    ```shell
    uv run python run_eval.py --baseline baseline_report.json
    ```

4. To add a case, add an entry to `wfo_search_dataset.yaml` with the question, the expected tool calls and the
   evaluators, and refer to the seed data in `docker/core/seed.py`. Then restore the canonical form of the file, as
   described in [Evals](../reference/evals.md#dataset).
5. To compare models, run the benchmark once for each `AGENT_MODEL` and compare the reports.
6. To gate CI, add `--fail-under 0.7`, or the floor of your choice. The workflow of the agent repository does this
   nightly and on pull requests labelled `eval`.
7. When a change is meant to move the numbers, bless a new baseline in the same change:

    ```shell
    uv run python run_eval.py --save-report baseline_report.json
    ```

## Verify it works

Open the report printed by `run_eval.py`. Each case shows its score per evaluator, and with `--baseline` the cases that
regressed are listed. A run without regressions exits with status 0.

## Reference

- [Evals](../reference/evals.md)
- [How we evaluate the agent](../architecture/evaluation.md)
