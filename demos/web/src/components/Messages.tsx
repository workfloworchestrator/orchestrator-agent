import type { Message, ToolCall } from "@ag-ui/core";
import { useEffect, useRef } from "react";
import { Check, Download } from "lucide-react";

import { artifactOf, textOf } from "../lib/artifacts";
import { Chart } from "./Chart";
import { Markdown } from "./Markdown";
import { CHIP_OFF } from "./classes";

/** The rail's line, drawn by each step: from its top edge to its node, and from its node down to the next step. */
const LINE_ABOVE =
  "before:absolute before:top-0 before:left-[7.5px] before:h-2.5 before:w-px before:origin-top before:bg-wfo-line";
const LINE_BELOW =
  "after:absolute after:top-2.5 after:bottom-0 after:left-[7.5px] after:w-px after:origin-top after:bg-wfo-line";

/**
 * One step the agent took, a node on the rail: the tool it called, and what came back.
 *
 * A step that arrives while the agent works is seen arriving: the line draws down to it, its row comes in
 * behind, and the check and the result's chip land when the result does. A stored chat's steps just are.
 */
function Step({
  call,
  result,
  running,
  fresh,
  first,
  last,
}: {
  call: ToolCall;
  result: string | undefined;
  running: boolean;
  /** Arrived during this run, rather than loaded with the chat. */
  fresh: boolean;
  first: boolean;
  last: boolean;
}) {
  const artifact = artifactOf(result);
  const name = call.function.name.replaceAll("_", " ");
  const done = result !== undefined;
  const chart = artifact?.chart;
  const table = !chart && artifact?.rendered_block?.type === "markdown" ? artifact.rendered_block.content : null;
  const above = first ? "" : `${LINE_ABOVE} ${fresh ? "motion-safe:before:animate-wfo-draw" : ""}`;
  const below = last ? "" : `${LINE_BELOW} ${fresh ? "motion-safe:after:animate-wfo-draw" : ""}`;
  const rise = fresh ? "motion-safe:animate-wfo-rise motion-safe:[animation-delay:160ms]" : "";
  const land = fresh ? "motion-safe:animate-wfo-land" : "";
  return (
    <li className={`relative flex flex-col gap-1.5 py-0.5 ${above} ${below}`}>
      <div className={`flex flex-wrap items-center gap-x-2 gap-y-1 text-xs leading-4 text-wfo-muted ${rise}`}>
        <span className="relative z-[1] inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-wfo-bg">
          <span
            key={done ? "done" : "waiting"}
            className={`inline-flex h-4 w-4 items-center justify-center rounded-full ${
              done ? `bg-wfo-accent-soft text-wfo-accent ${land}` : "border border-wfo-line bg-wfo-bg"
            }`}
          >
            {done ? (
              <Check size={10} strokeWidth={3} />
            ) : running ? (
              <span className="h-1.5 w-1.5 rounded-full bg-wfo-muted motion-safe:animate-wfo-blink" />
            ) : null}
          </span>
        </span>
        <span>{name}</span>
        {artifact && (
          <span
            className={`inline-flex items-center gap-1.5 rounded-full bg-wfo-surface px-2 py-0.5 text-wfo-ink ${land}`}
          >
            {artifact.description ?? artifact.entity_type ?? "result"}
            {artifact.download_url && (
              <a
                href={artifact.download_url}
                target="_blank"
                rel="noreferrer"
                title="Download"
                className="inline-flex text-wfo-accent"
              >
                <Download size={13} />
              </a>
            )}
          </span>
        )}
      </div>
      {chart && (
        <div className={`ml-6 ${fresh ? "motion-safe:animate-wfo-rise" : ""}`}>
          <Chart chart={chart} />
        </div>
      )}
      {table && (
        <div
          className={`ml-6 overflow-x-auto rounded-lg border border-wfo-line bg-wfo-bg px-3 py-2 ${fresh ? "motion-safe:animate-wfo-rise" : ""}`}
        >
          <Markdown text={table} />
        </div>
      )}
    </li>
  );
}

type Item =
  | { kind: "user"; id: string; text: string }
  | { kind: "text"; id: string; text: string }
  | { kind: "steps"; id: string; steps: { call: ToolCall; result: string | undefined }[] };

function Typing() {
  return (
    <div className="flex items-center gap-1 self-start px-1 py-2" aria-label="Working">
      <span className="h-1.5 w-1.5 rounded-full bg-wfo-muted motion-safe:animate-wfo-blink" />
      <span className="h-1.5 w-1.5 rounded-full bg-wfo-muted [animation-delay:150ms] motion-safe:animate-wfo-blink" />
      <span className="h-1.5 w-1.5 rounded-full bg-wfo-muted [animation-delay:300ms] motion-safe:animate-wfo-blink" />
    </div>
  );
}

export function Messages({
  messages,
  running,
  suggestions,
  onSuggest,
}: {
  messages: Message[];
  running: boolean;
  suggestions: string[];
  onSuggest: (text: string) => void;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-col gap-3 pt-1">
        <p className="m-0 text-sm leading-6 text-wfo-ink">
          Ask about subscriptions, products and processes, or start a workflow. Its form is filled in here, page by
          page, and nothing starts until you approve.
        </p>
        {suggestions.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {suggestions.map((text) => (
              <button key={text} type="button" className={CHIP_OFF} onClick={() => onSuggest(text)}>
                {text}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  // A tool's result is its own message, keyed on the call it answers.
  const results = new Map<string, string>();
  for (const m of messages) if (m.role === "tool") results.set(m.toolCallId, textOf(m.content));
  const last = messages[messages.length - 1];
  const thinking = running && (last.role !== "assistant" || !textOf(last.content));

  // What is shown, in order: the person's messages, the agent's words, and its steps — consecutive steps
  // joined on one rail even when the model took them in separate turns.
  const items: Item[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      items.push({ kind: "user", id: m.id, text: textOf(m.content) });
    } else if (m.role === "assistant") {
      const text = textOf(m.content);
      if (text) items.push({ kind: "text", id: m.id, text });
      for (const call of m.toolCalls ?? []) {
        const previous = items[items.length - 1];
        const step = { call, result: results.get(call.id) };
        if (previous?.kind === "steps") previous.steps.push(step);
        else items.push({ kind: "steps", id: call.id, steps: [step] });
      }
    }
    // tool results are shown under their call; system and other roles are not shown
  }

  return (
    <div className="flex flex-col gap-3">
      {items.map((item) => {
        if (item.kind === "user") {
          return (
            <div
              key={item.id}
              className="max-w-[85%] self-end rounded-2xl rounded-br-md bg-wfo-accent px-3.5 py-2 text-sm leading-5 whitespace-pre-wrap text-wfo-on-accent"
            >
              {item.text}
            </div>
          );
        }
        if (item.kind === "text") {
          return (
            <div key={item.id} className="max-w-full self-start text-sm text-wfo-ink">
              <Markdown text={item.text} />
            </div>
          );
        }
        return (
          <ol key={item.id} className="m-0 flex max-w-full list-none flex-col self-start p-0">
            {item.steps.map(({ call, result }, index) => (
              <Step
                key={call.id}
                call={call}
                result={result}
                running={running}
                fresh={running}
                first={index === 0}
                last={index === item.steps.length - 1}
              />
            ))}
          </ol>
        );
      })}
      {thinking && <Typing />}
      <div ref={bottom} />
    </div>
  );
}
