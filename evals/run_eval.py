"""Run the WFO search agent golden-set benchmark with pydantic-evals.

The agent runs in-process — built by the same ``build_agent`` assembly the
server adapters use — against the live orchestrator-core MCP server from
docker-compose.yml, once per model in ../eval-models.json whose provider key
is set. The built-in span-based evaluators (TrajectoryMatch,
ArgumentCorrectness, MaxToolCalls) read the tool calls from the OpenTelemetry
spans pydantic-ai emits, so this measures agent behavior only: which tools,
in what order, with which arguments.

    uv run python run_eval.py [--models openai:gpt-4o-mini] [--cases name1,name2] [--repeat 5] [--fail-under 0.8]
"""

import argparse
import json
import os
import sys
from collections.abc import Awaitable, Callable
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from settings import eval_settings

# The agent's AgentSettings reads the process environment at import time, so
# the resolved eval settings must be published before importing the agent.
eval_settings.apply_to_environment()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# The span-based evaluators need a real (SDK) tracer provider; pydantic-evals
# attaches its in-memory exporter to it per case.
trace.set_tracer_provider(TracerProvider())

from pydantic_ai import Agent

Agent.instrument_all()

from pydantic_ai.models import Model
from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter
from rich.align import Align
from rich.console import Console
from rich.table import Table

from orchestrator_agent.agent import WFOAgent, build_agent, new_deps


def build_model(name: str, api_key: str) -> Model:
    """The pydantic-ai model for a ``provider:model`` name, with that provider's key."""
    provider, _, model_name = name.partition(":")
    if provider == "openai":
        # Chat Completions, as the agent's own settings build it; the plain model string would take the
        # Responses API, which the locked openai SDK cannot read.
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key))
    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # The instructions and tool definitions are the same on every call: cache them.
        settings = AnthropicModelSettings(anthropic_cache_instructions=True, anthropic_cache_tool_definitions=True)
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_key), settings=settings)
    raise SystemExit(f"{name}: no provider {provider!r} in build_model")


def task_for(agent: WFOAgent) -> Callable[[str], Awaitable[str]]:
    """The task under evaluation: one user turn of this agent against the real MCP toolset."""

    async def run_wfo_agent(question: str) -> str:
        deps = new_deps(question)
        async with agent:
            result = await agent.run(question, deps=deps)
        return result.output

    return run_wfo_agent


def for_model(path: Path, model: str) -> Path:
    """``baseline_report.json`` for ``openai:gpt-4o-mini`` is ``baseline_report.openai-gpt-4o-mini.json``."""
    return path.with_name(f"{path.stem}.{model.replace(':', '-').replace('/', '-')}{path.suffix}")


def case_means(report: EvaluationReport) -> dict[str, dict[str, float]]:
    """Per case, the average of every score and the pass rate of every assertion over the case's runs.

    A ``--repeat`` report holds one entry per run of a case, which pydantic-evals
    groups back by case; without repeats every case is its own single run.
    """
    groups = report.case_groups()
    runs_by_case = {case.name: [case] for case in report.cases} if groups is None else {g.name: g.runs for g in groups}
    means: dict[str, dict[str, float]] = {}
    for case_name, runs in runs_by_case.items():
        values: defaultdict[str, list[float]] = defaultdict(list)
        for run in runs:
            for name, result in (run.scores | run.assertions).items():
                values[name].append(float(result.value))
        means[case_name] = {name: fmean(of_runs) for name, of_runs in values.items()}
    return means


def route_and_arguments(report: EvaluationReport) -> dict[str, tuple[float, float | None]]:
    """Per case over its runs: the route score average and the argument-correctness pass rate.

    A run that crashed scores 0 on both; a case without an argument assertion has no pass rate.
    """
    runs: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    for case in report.cases:
        arguments = case.assertions.get("ArgumentCorrectness")
        route = case.scores["TrajectoryMatch"].value
        runs[case.source_case_name or case.name].append(
            (float(route), None if arguments is None else float(arguments.value))
        )
    for failure in report.failures:
        runs[failure.source_case_name or failure.name].append((0.0, 0.0))
    return {
        name: (
            fmean(route for route, _ in pairs),
            fmean(rates) if (rates := [a for _, a in pairs if a is not None]) else None,
        )
        for name, pairs in runs.items()
    }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_table(reports: dict[str, EvaluationReport | None], cases: list[str], repeat: int) -> None:
    """One table: a column per model holding two sub-columns, route match and argument correctness, a row per case.

    rich tables cannot span a header over two columns, so each model's cell holds a two-column grid; every grid has
    the same rows (a header line, the cases, the averages), so the rows line up across the models.
    """
    scores = {model: route_and_arguments(report) for model, report in reports.items() if report is not None}

    def column(model: str) -> Table:
        grid = Table.grid(padding=(0, 2))
        if model not in scores:
            grid.add_column()
            grid.add_row("no key")
            return grid
        grid.add_column(justify="right")
        grid.add_column(justify="right")
        grid.add_row("route", "arguments")
        for case in cases:
            grid.add_row(*(_percent(value) for value in scores[model][case]))
        routes = [route for route, _ in scores[model].values()]
        arguments = [a for _, a in scores[model].values() if a is not None]
        grid.add_row(_percent(fmean(routes)), _percent(fmean(arguments) if arguments else None), style="bold")
        return grid

    names = Table.grid()
    names.add_column(no_wrap=True)
    names.add_row("")  # level with the route / arguments line
    for case in cases:
        names.add_row(case)
    names.add_row("Averages", style="bold")

    table = Table(title=f"Average over {repeat} run(s) per case")
    table.add_column("Case")
    for model in reports:
        table.add_column(model, justify="center")
    table.add_row(names, *(Align.center(column(model)) for model in reports))
    # Wide enough for every column, so a narrow terminal or a CI log wraps rather than truncates.
    Console(width=max(Console().width, 28 + 30 * len(reports))).print(table)


def compare_to_baseline(report: EvaluationReport, baseline: EvaluationReport, tolerance: float) -> list[str]:
    """Per-case regressions of this run against a stored baseline report.

    Compares every score average and assertion pass rate of every case present
    in both reports: either counts as regressed when it drops more than
    ``tolerance`` below the baseline (LLM runs are non-deterministic; the
    tolerance absorbs run-to-run noise). With one run per case a pass rate is 1
    or 0, so an assertion that flips from passing to failing is a regression.
    Improvements and cases only present on one side never fail the comparison.
    """
    baseline_means = case_means(baseline)
    regressions: list[str] = []
    for case_name, means in case_means(report).items():
        base = baseline_means.get(case_name)
        if base is None:
            print(f"baseline: case {case_name!r} not in baseline, skipping")
            continue
        for name, value in means.items():
            if name in base and value - base[name] < -tolerance:
                regressions.append(f"{case_name} :: {name} {base[name]:.2f} -> {value:.2f}")
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        help="comma-separated models from eval-models.json to run (default: every one whose provider key is set)",
    )
    parser.add_argument("--cases", help="comma-separated case names to run (default: all)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run every case this many times; scores and assertions then count as per-case averages",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="how many runs go at once; the agent shares one MCP session and core's tools keep no session state",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero when the mean over all assertions/scores drops below this",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="stored report (see --save-report) to compare against, one file per model; exit non-zero on per-case regression",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="score drop vs the baseline that still counts as run-to-run noise (default 0.2)",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="write each model's full report as JSON, the model in the file name (baseline_report.json blesses baselines)",
    )
    args = parser.parse_args()

    dataset = Dataset[str, str].from_file(eval_settings.dataset_file)
    if args.cases:
        wanted = {name.strip() for name in args.cases.split(",")}
        dataset = Dataset[str, str](
            name=dataset.name,
            cases=[c for c in dataset.cases if c.name in wanted],
            evaluators=dataset.evaluators,
        )

    models: dict[str, str] = json.loads(eval_settings.models_file.read_text())
    if args.models:
        selected = [name.strip() for name in args.models.split(",")]
        if unknown := [name for name in selected if name not in models]:
            sys.exit(f"not in {eval_settings.models_file.name}: {', '.join(unknown)}")
        models = {name: models[name] for name in selected}

    failures: list[str] = []
    reports: dict[str, EvaluationReport | None] = {}
    for model_name, key in models.items():
        if not os.environ.get(key):
            reports[model_name] = None  # shown as "no key"
            continue
        agent = build_agent(build_model(model_name, os.environ[key]))
        report = dataset.evaluate_sync(
            task_for(agent), name=model_name, max_concurrency=args.concurrency, repeat=args.repeat
        )
        reports[model_name] = report

        if args.save_report:
            path = for_model(args.save_report, model_name)
            path.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
            print(f"{model_name}: report written to {path}")

        if report.failures:
            failures.append(f"{model_name}: {len(report.failures)} case(s) failed to execute")
            continue

        if args.baseline:
            path = for_model(args.baseline, model_name)
            if not path.exists():
                print(f"{model_name}: no baseline at {path.name}, nothing to compare against")
            else:
                baseline = EvaluationReportAdapter.validate_json(path.read_bytes())
                regressions = compare_to_baseline(report, baseline, args.tolerance)
                for line in regressions:
                    print(f"REGRESSION vs {path.name}: {line}")
                if regressions:
                    failures.append(f"{model_name}: {len(regressions)} regression(s) vs baseline {path.name}")
                else:
                    print(f"{model_name}: no regressions vs baseline {path.name} (tolerance {args.tolerance})")

        if args.fail_under is not None:
            averages = report.averages()
            values = list(averages.scores.values())
            if averages.assertions is not None:
                values.append(averages.assertions)
            mean = sum(values) / len(values) if values else 0.0
            print(f"{model_name}: overall mean (score averages + assertion pass rate): {mean:.3f}")
            if mean < args.fail_under:
                failures.append(f"{model_name}: overall mean {mean:.3f} below --fail-under {args.fail_under}")

    print_table(reports, [case.name for case in dataset.cases], args.repeat)
    if not any(reports.values()):
        failures.append(f"no model ran: set a provider key from {eval_settings.models_file.name}")
    if failures:
        sys.exit("; ".join(failures))


if __name__ == "__main__":
    main()
