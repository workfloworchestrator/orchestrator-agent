import type { ChartData } from "../api/types";

export const number = new Intl.NumberFormat();
const month = new Intl.DateTimeFormat(undefined, { month: "short", year: "numeric" });
const day = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" });

/** Longer category labels than this crowd an axis. */
export const LONG_LABEL = 12;

/** A category as the reader says it: a date bucket as its month or day, anything else as it came. */
export function worded(label: string): string {
  const date = /^(\d{4})-(\d{2})-(\d{2})(?:[T ]00:00:00(?:\.0+)?(?:Z|[+-]00:?00)?)?$/.exec(label);
  if (!date) return label;
  const at = new Date(Date.UTC(Number(date[1]), Number(date[2]) - 1, Number(date[3])));
  return date[3] === "01" ? month.format(at) : day.format(at);
}

/** Many or long categories read better as bars going sideways, with every label written out. */
export function sideways(chart: ChartData): boolean {
  return (
    chart.kind !== "line" &&
    (chart.points.length > 8 || chart.points.some((point) => worded(point.label).length > LONG_LABEL))
  );
}
