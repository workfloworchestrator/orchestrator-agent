import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { ChartData } from "../api/types";
import { LONG_LABEL, number, sideways, worded } from "../lib/charts";

/** The theme's colours as SVG attributes need them: read off the element, so dark mode and a host's overrides count. */
export interface Ink {
  accent: string;
  soft: string;
  ink: string;
  muted: string;
  line: string;
  bg: string;
}

export function readInk(element: Element): Ink {
  const style = getComputedStyle(element);
  const of = (name: string) => style.getPropertyValue(name).trim();
  return {
    accent: of("--color-wfo-accent"),
    soft: of("--color-wfo-accent-soft"),
    ink: of("--color-wfo-ink"),
    muted: of("--color-wfo-muted"),
    line: of("--color-wfo-line"),
    bg: of("--color-wfo-bg"),
  };
}

const shortened = (label: string) => {
  const text = worded(label);
  return text.length > LONG_LABEL ? `${text.slice(0, LONG_LABEL - 1)}…` : text;
};

/**
 * The chart itself. Magnitudes are bars in the one accent hue (core's pie hint included: a part-to-whole
 * reads better as bars than as slices), a series over time is a line; grid and axes stay recessive.
 */
export function Plot({ chart, ink, height, full }: { chart: ChartData; ink: Ink; height: number; full: boolean }) {
  const tick = { fill: ink.muted, fontSize: 11 };
  const count = (value: unknown) => number.format(Number(value));
  const tooltip = (
    <Tooltip
      cursor={chart.kind === "line" ? { stroke: ink.line } : { fill: ink.soft }}
      formatter={(value) => [count(value), chart.metric]}
      labelFormatter={(label) => worded(String(label))}
      contentStyle={{
        background: ink.bg,
        border: `1px solid ${ink.line}`,
        borderRadius: 8,
        color: ink.ink,
        fontSize: 12,
        padding: "6px 10px",
      }}
      labelStyle={{ color: ink.ink, fontWeight: 600 }}
      itemStyle={{ color: ink.muted }}
    />
  );
  if (chart.kind === "line") {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chart.points} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={ink.line} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            tick={tick}
            tickLine={false}
            axisLine={{ stroke: ink.line }}
            interval="preserveStartEnd"
            minTickGap={24}
            tickFormatter={full ? worded : shortened}
          />
          <YAxis tick={tick} tickLine={false} axisLine={false} width={40} tickFormatter={count} />
          {tooltip}
          <Line
            type="monotone"
            dataKey="value"
            stroke={ink.accent}
            strokeWidth={2}
            dot={{ r: 3, fill: ink.accent, strokeWidth: 0 }}
            activeDot={{ r: 5, fill: ink.accent, stroke: ink.bg, strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }
  if (full && sideways(chart)) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={chart.points}
          layout="vertical"
          margin={{ top: 8, right: 16, bottom: 0, left: 8 }}
          barCategoryGap={2}
        >
          <CartesianGrid stroke={ink.line} strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={tick} tickLine={false} axisLine={false} tickFormatter={count} />
          <YAxis
            type="category"
            dataKey="label"
            tick={tick}
            tickLine={false}
            axisLine={false}
            width={180}
            interval={0}
            tickFormatter={worded}
          />
          {tooltip}
          <Bar dataKey="value" fill={ink.accent} radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chart.points} margin={{ top: 8, right: 12, bottom: 0, left: 0 }} barCategoryGap={2}>
        <CartesianGrid stroke={ink.line} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="label"
          tick={tick}
          tickLine={false}
          axisLine={{ stroke: ink.line }}
          interval={full ? 0 : "preserveStartEnd"}
          minTickGap={16}
          tickFormatter={shortened}
        />
        <YAxis tick={tick} tickLine={false} axisLine={false} width={40} tickFormatter={count} />
        {tooltip}
        <Bar dataKey="value" fill={ink.accent} radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
