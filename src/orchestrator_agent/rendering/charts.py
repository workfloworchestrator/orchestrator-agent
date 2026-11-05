# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic Mermaid rendering for aggregate results.

A grouped aggregation is always a distribution, so it should always be charted — but
prompt-driven rendering is unreliable (the model tables some results and charts others).
This module builds the ```mermaid block straight from the ``aggregate`` tool result instead,
using core's own ``visualization`` hint to pick the diagram type. Adapters inject the result
into the answer when the model didn't already include a chart, so charting is guaranteed for
every text-only client (kagent, native LibreChat) regardless of model behaviour.

The chart is first read as data (``aggregate_to_chart``: a ``ChartData`` a client such as the popup draws
itself) from the ``AggregateToolResponse`` JSON shape: ``{visualization: str, results: [{group_values:
{...}, aggregations: {...}}]}``; ``chart_to_mermaid`` renders that data, and ``aggregate_to_mermaid`` is
the two together. All return ``None`` for ungrouped/scalar results (one number, no categories — nothing
to chart).
"""

from __future__ import annotations

from typing import Any, Literal

from orchestrator_agent.artifacts import ChartData, ChartPoint


def aggregate_to_mermaid(payload: Any) -> str | None:
    """Render an ``aggregate`` tool result as Mermaid diagram source, or ``None`` when it is not a chart."""
    chart = aggregate_to_chart(payload)
    return chart_to_mermaid(chart) if chart is not None else None


def aggregate_to_chart(payload: Any) -> ChartData | None:
    """Read an ``aggregate`` tool result as the chart's data, or ``None``.

    Returns ``None`` when the result is not a grouped distribution (no rows, or a scalar
    aggregate with no ``group_values`` — a single number that no chart would clarify).
    """
    if not isinstance(payload, dict):
        return None
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        return None

    pairs: list[tuple[str, float | int]] = []
    grouped = False
    metric = "value"
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_values = row.get("group_values") or {}
        if group_values:
            grouped = True
        picked = _pick_aggregation(row.get("aggregations") or {})
        if picked is None:
            continue
        metric, value = picked
        pairs.append((_label(group_values), value))

    if not grouped or not pairs:
        return None  # scalar / no categories — not a chart.

    group_keys = _group_key_names(rows)
    title = f"{metric} by {group_keys}" if group_keys else metric
    visualization = str(payload.get("visualization", "")).lower()
    kind: Literal["pie", "bar", "line"] = (
        "pie" if visualization == "pie" else "line" if visualization == "line" else "bar"
    )
    points = [ChartPoint(label=label, value=value) for label, value in pairs]
    return ChartData(kind=kind, title=title, metric=metric, points=points)


def chart_to_mermaid(chart: ChartData) -> str:
    """The chart's data as Mermaid diagram source: a pie, or an xychart with bars or a line."""
    pairs = [(point.label, point.value) for point in chart.points]
    if chart.kind == "pie":
        return _pie(chart.title, pairs)
    return _xychart(chart.title, chart.metric, pairs, chart.kind)


# --- helpers -------------------------------------------------------------------------------


def _label(group_values: dict[str, Any]) -> str:
    """Human label for a row: the group value(s), joined when grouped on multiple fields."""
    if not group_values:
        return "(all)"
    return " / ".join(str(v) for v in group_values.values())


def _pick_aggregation(aggregations: dict[str, Any]) -> tuple[str, float | int] | None:
    """The ``(metric_name, value)`` to plot — the sole aggregation, else a count, else the first."""
    if not aggregations:
        return None
    if len(aggregations) == 1:
        return next(iter(aggregations.items()))
    for key, value in aggregations.items():
        if key.lower().startswith("count"):
            return key, value
    return next(iter(aggregations.items()))


def _group_key_names(rows: list[Any]) -> str:
    """Distinct group-by field names across all rows, for the chart title."""
    keys: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            for key in row.get("group_values") or {}:
                if key not in keys:
                    keys.append(key)
    return ", ".join(keys)


def _fmt_num(value: float | int) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _quote(text: Any) -> str:
    """Quote a Mermaid string literal, neutralising embedded double quotes."""
    return '"' + str(text).replace('"', "'") + '"'


def _pie(title: str, pairs: list[tuple[str, float | int]]) -> str:
    lines = [f"pie title {title}"]
    lines += [f"    {_quote(label)} : {_fmt_num(value)}" for label, value in pairs]
    return "\n".join(lines)


def _xychart(title: str, metric: str, pairs: list[tuple[str, float | int]], series: str) -> str:
    labels = ", ".join(_quote(label) for label, _ in pairs)
    values = ", ".join(_fmt_num(value) for _, value in pairs)
    lines = [
        "xychart-beta",
        f"    title {_quote(title)}",
        f"    x-axis [{labels}]",
        f"    y-axis {_quote(metric)}",
        f"    {series} [{values}]",
    ]
    return "\n".join(lines)


__all__ = ["aggregate_to_chart", "aggregate_to_mermaid", "chart_to_mermaid"]
