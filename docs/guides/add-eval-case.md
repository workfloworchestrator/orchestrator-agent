# How to add an evaluation case

Add a real request or an observed failure to the benchmark so you can check it on later changes.

## Before you start

Have a locally running Orchestrator Agent, as described in [Getting started](../getting-started.md).
Prepare the [evaluation environment](evals.md#1-prepare-the-evaluation-environment) if you have not already done so.
Work from the checkout's `evals/` directory.

## 1. Define the behavior to check

Choose the request you want to protect. Write down what the agent should do and what it should avoid. Check the
available data in `docker/core/seed.py`. If the request needs other data, extend the seed and recreate the benchmark
stack with a fresh database before running the case.

Choose checks for those requirements from the [evaluator reference](../reference/evals.md#evaluators). Use an exact
sequence only when you need to reject other routes. Include a negative case when you also need to check that the
action is avoided.

## 2. Add the case to the dataset

Append an item to the `cases` list in `wfo_search_dataset.yaml`. Give it a unique name, enter the request in `inputs`,
and record the expected behavior in `metadata.curation_note`.

For a product-count request, use:

```yaml
- name: count-products
  inputs: How many products are there in total?
  metadata:
    failure_modes:
    - aggregation
    curation_note: A total product count needs aggregate with PRODUCT, not subscription search.
  expected_output: null
  evaluators:
  - TrajectoryMatch:
      expected_trajectory:
      - aggregate
      order: exact
  - ArgumentCorrectness:
      tool_name: aggregate
      expected_arguments:
        entity_type: PRODUCT
      match_mode: subset
```

Adapt the request and evaluators to your task. Keep the existing dataset-wide `MaxToolCalls` evaluator. If you need to
verify the final answer automatically, add an evaluator for that requirement as well; see
[Tool use, answers and completed tasks](../explanation/evaluation.md#tool-use-answers-and-completed-tasks).

## 3. Validate and run the case

Restore the dataset's canonical serialization and refresh its schema:

```shell
uv run python -c "from pydantic_evals import Dataset; \
  Dataset[str, str].from_file('wfo_search_dataset.yaml').to_file('wfo_search_dataset.yaml')"
```

Run the case by its name, substituting yours if different:

```shell
uv run python run_eval.py --cases count-products --save-report count-products.local.json
```

Confirm that `count-products.local.json` contains the named case. Inspect its scores, assertions, reasons and final
output, even when the command exits successfully. For the product-count example, manually verify that the answer
gives three products. Use the [report reference](../reference/evals.md#reports) to interpret the results.

If execution fails, resolve the reported exception and rerun. If a check fails, inspect the behavior and its expected
route; fix the agent or correct an expectation that rejects a valid solution.

## 4. Review the extended dataset

Run the full dataset and inspect the results:

```shell
uv run python run_eval.py --save-report extended.local.json
```

Confirm that the new case and the existing cases are present. Review any new failures and include the dataset,
generated schema and seed changes together for review. If the extended dataset needs a new baseline, follow
[How to benchmark prompt and model changes](evals.md#2-capture-or-select-a-baseline) using the agent configuration
you intend to use as the reference.

## Related pages

- [How to benchmark prompt and model changes](evals.md)
- [Evals reference](../reference/evals.md)
- [Understanding agent evaluations](../explanation/evaluation.md)
