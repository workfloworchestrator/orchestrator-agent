# How to benchmark prompt and model changes

Compare a prompt or model change against a reviewed baseline before using it.

## Before you start

Have a locally running Orchestrator Agent. Follow [Getting started](../getting-started.md) if you have not set it up yet.
Choose an existing dataset that covers the requests you want to compare. To extend it first, follow
[How to add an evaluation case](add-eval-case.md).

## 1. Prepare the evaluation environment

From your existing agent checkout:

```shell
cd evals
uv sync --locked
```

Create `evals/.env` from `evals/.env.example` if it does not already exist. Reuse your working model and credentials.
Export any additional proxy, Azure or domain-context settings as listed in [Eval settings](../reference/evals.md#settings).

Use the bundled seeded Core and database for these cases. Make host ports 5433 and 8086 available, then start them:

```shell
docker compose -f docker-compose.yml up -d --wait
```

Keep working from `evals/` for the following steps.

## 2. Capture or select a baseline

Before changing the agent, capture a full run:

```shell
uv run python run_eval.py --save-report before.local.json
```

Review each case's scores, assertions, reasons and final output in `before.local.json` before accepting the report as
your baseline. Resolve execution errors before continuing. Use the [report reference](../reference/evals.md#reports)
to interpret the results.

Record the Git revision, local edits, dataset, Core image, model identifier, provider endpoint and domain context
alongside the report. If you already have a reviewed baseline for this configuration and dataset, use that report
instead of capturing another.

## 3. Make one change and compare it

Change one thing: a prompt, a plugin, or the model. Keep the dataset, case names, evaluators and backing environment
fixed. For a prompt comparison, keep the model fixed. For a model comparison, change `AGENT_MODEL` and keep the agent
code fixed.

Run the candidate and save its results separately, substituting your chosen baseline if needed:

```shell
uv run python run_eval.py --baseline before.local.json --save-report after.local.json --fail-under 0.7
```

Choose a score floor appropriate to your requirements; adjust `--fail-under` accordingly. For gate definitions and
regression tolerance, see [Commands](../reference/evals.md#commands) and [Baseline](../reference/evals.md#baseline).

Check that both full reports contain the same cases and checks. If you change the dataset or evaluators during the
comparison, rerun the original agent configuration to capture a matching baseline before comparing again.

## 4. Resolve regressions and accept the change

For each reported regression, inspect its reason and output in `after.local.json`. Fix incorrect behavior or revise an
expectation that rejects a valid solution, then rerun the comparison. Rerun uncertain cases with `--cases`, saving each
attempt separately. For a full trace when reproducing a failure, follow [How to trace the agent](tracing.md).

Before accepting the change, review the full candidate report and verify any important answer or behavior outside
its checks. If the accepted change should establish a new committed baseline, run:

```shell
uv run python run_eval.py --save-report baseline_report.json
```

Review this run too, and include the baseline diff with the change. Do not overwrite a baseline just to clear a
regression. Use the existing [eval CI workflow](../reference/evals.md#ci) to check the committed baseline and score floor.

## Related pages

- [How to add an evaluation case](add-eval-case.md)
- [Evals reference](../reference/evals.md)
- [Understanding agent evaluations](../architecture/evaluation.md)
