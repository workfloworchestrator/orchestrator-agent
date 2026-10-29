# Evals

The `evals/` folder of the agent repository holds a golden-set benchmark built on
[pydantic-evals](https://ai.pydantic.dev/evals/). The agent runs in-process against a live Orchestrator Core MCP
server, so the benchmark measures agent behaviour only: which tools are called, in which order and with which
arguments. See [How we evaluate the agent](../architecture/evaluation.md) for the reasoning and
[Benchmark prompt and model changes](../guides/evals.md) for a walkthrough.

## Dataset

`wfo_search_dataset.yaml` is a hand-curated set of cases in the native pydantic-evals schema. Each case pairs a
user question with evaluators. Cases refer to the deterministic seed data in `docker/core/seed.py`.
`metadata.failure_modes` tags the shape of the task and `metadata.curation_note` explains why the expected
trajectory is the ideal one.

The file uses the canonical serialized form, and `wfo_search_dataset_schema.json` provides IDE validation. After
editing cases by hand, round-trip the file to restore the canonical form and refresh the schema:

```shell
uv run python -c "from pydantic_evals import Dataset; \
  Dataset[str, str].from_file('wfo_search_dataset.yaml').to_file('wfo_search_dataset.yaml')"
```

## Evaluators

| Evaluator | What it asserts |
| --- | --- |
| `TrajectoryMatch(order='exact')` | The tool-call sequence, exactly. Used where one route is right, including negative examples with an empty expected trajectory |
| `TrajectoryMatch(order='in_order')` | F1 over the longest common subsequence, so a rule-consistent detour earns partial credit |
| `ArgumentCorrectness(match_mode='subset')` | The arguments of one call that the prompt influences, such as the entity type and identifiers passed through verbatim |
| `MaxToolCalls(8)` | Dataset-wide call budget |

Tool calls are read from the OpenTelemetry spans that pydantic-ai emits.

## Commands

Start the backing services, an Orchestrator Core with the MCP server and a seeded database, then run from `evals/`:

```shell
docker compose -f docker-compose.yml up -d --wait
uv sync
cp .env.example .env   # set AGENT_API_KEY and pin AGENT_MODEL
```

| Command | Effect |
| --- | --- |
| `uv run python run_eval.py` | Run the full golden set |
| `uv run python run_eval.py --cases <name>` | Run a subset, for a smoke test |
| `uv run python run_eval.py --fail-under 0.7` | Exit non-zero when the score is below the floor |
| `uv run python run_eval.py --baseline baseline_report.json` | Fail when a case regresses against the committed baseline |
| `uv run python run_eval.py --save-report baseline_report.json` | Bless a new baseline |

Scores are only comparable on the same model, so pin `AGENT_MODEL`.

### Baseline

`baseline_report.json` is a full report of a blessed run. A score that drops more than `--tolerance` (default 0.2)
below the baseline, or an assertion that flips from passing to failing, fails the run and names the case.
Improvements never fail. Bless a new baseline in the same change that moves the numbers, so a baseline change is
always a reviewed diff.

### CI

`.github/workflows/eval.yml` runs the benchmark on demand, nightly and on pull requests labelled `eval`. It gates on
the committed baseline and the `--fail-under` floor. It needs the `EVAL_AGENT_API_KEY` repository secret and
optionally an `EVAL_AGENT_MODEL` variable (default `openai:gpt-4o`).
