import { type MouseEvent, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Maximize2, X } from "lucide-react";

import type { ChartData } from "../api/types";
import { number, sideways, worded } from "../lib/charts";
import { ICON_BUTTON } from "./classes";
import { type Ink, Plot, readInk } from "./Plot";

const CELL = "border-0 border-b border-wfo-line px-2 py-1 text-left";

/** A chart the agent drew for a result: small in the chat, full size with its table on a click. */
export function Chart({ chart }: { chart: ChartData }) {
  const root = useRef<HTMLDivElement>(null);
  const [ink, setInk] = useState<Ink | null>(null);
  const [full, setFull] = useState(false);

  useLayoutEffect(() => {
    const element = root.current;
    if (!element) return;
    const read = () => setInk(readInk(element));
    read();
    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    scheme.addEventListener("change", read);
    return () => scheme.removeEventListener("change", read);
  }, []);
  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFull(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);

  const groupedBy = chart.title.includes(" by ") ? chart.title.slice(chart.title.indexOf(" by ") + 4) : "category";
  const fullHeight = sideways(chart) ? Math.min(Math.max(chart.points.length * 26 + 40, 240), 640) : 400;
  const stay = (e: MouseEvent) => e.stopPropagation();

  return (
    <div ref={root} className="overflow-hidden rounded-lg border border-wfo-line bg-wfo-bg">
      <div className="flex items-center gap-2 pt-1.5 pr-1.5 pl-3">
        <span className="min-w-0 flex-1 truncate text-xs leading-4 font-medium text-wfo-ink">{chart.title}</span>
        <button type="button" className={`${ICON_BUTTON} h-6 w-6`} title="Open full size" onClick={() => setFull(true)}>
          <Maximize2 size={14} />
        </button>
      </div>
      <div className="cursor-zoom-in" onClick={() => setFull(true)} title="Open full size">
        {ink && <Plot chart={chart} ink={ink} height={180} full={false} />}
      </div>

      {full && ink && (
        <div
          className="fixed inset-0 z-10 flex items-center justify-center bg-wfo-ink/40 p-4"
          onClick={() => setFull(false)}
        >
          <div
            role="dialog"
            aria-label={chart.title}
            onClick={stay}
            className="flex max-h-full w-[min(960px,100%)] flex-col overflow-hidden rounded-2xl bg-wfo-bg shadow-wfo-panel"
          >
            <div className="flex items-center gap-2 border-b border-wfo-line px-4 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm leading-5 font-semibold">{chart.title}</div>
                <div className="text-xs leading-4 text-wfo-muted">
                  {chart.metric} for each {groupedBy}
                </div>
              </div>
              <button type="button" className={ICON_BUTTON} title="Close" onClick={() => setFull(false)}>
                <X size={18} />
              </button>
            </div>
            <div className="[scrollbar-width:thin] overflow-y-auto p-4">
              <Plot chart={chart} ink={ink} height={fullHeight} full />
              <table className="mt-4 w-full border-collapse text-xs">
                <thead>
                  <tr>
                    <th className={`${CELL} font-semibold text-wfo-muted`}>{groupedBy}</th>
                    <th className={`${CELL} text-right font-semibold text-wfo-muted`}>{chart.metric}</th>
                  </tr>
                </thead>
                <tbody className="[&>tr:last-child>td]:border-b-0">
                  {chart.points.map((point, index) => (
                    <tr key={index}>
                      <td className={CELL}>{worded(point.label)}</td>
                      <td className={`${CELL} text-right tabular-nums`}>{number.format(point.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
