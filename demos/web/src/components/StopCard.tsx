import type { Interrupt } from "@ag-ui/core";
import { useState } from "react";
import { FileText } from "lucide-react";

import type { ApprovalAnswer, ApprovalStop, AskAnswers, AskedQuestion, AskStop, FieldOption } from "../api/types";
import { Combobox } from "./Combobox";
import { Markdown } from "./Markdown";
import { CHIP_OFF, CHIP_ON, FIELD, GHOST, PRIMARY } from "./classes";

/** What a stop carries in its metadata: a form page's questions, or the start to approve. */
type Stop = Partial<AskStop & ApprovalStop>;

/** The answer to one question: the positions of the choices picked, the text typed, or the options found by search. */
type Answer = number[] | string | FieldOption[];

/** A search for a question's options, as the stop card gets it from the chat. */
export type Search = (field: string, words: string) => Promise<FieldOption[]>;

/** The positions picked among a question's choices; none for any other kind of answer. */
function positions(answer: Answer | undefined): number[] {
  return Array.isArray(answer) ? answer.filter((item): item is number => typeof item === "number") : [];
}

/** The options found by search; none for any other kind of answer. */
function found(answer: Answer | undefined): FieldOption[] {
  return Array.isArray(answer) ? answer.filter((item): item is FieldOption => typeof item === "object") : [];
}

function picks(answer: Answer | undefined): (number | string)[] {
  if (answer === undefined) return [];
  if (typeof answer === "string") return answer.trim() ? [answer.trim()] : [];
  // A searched option is answered with its value, which the agent resolves exactly.
  return answer.map((item) => (typeof item === "object" ? item.value : item));
}

function Choices({
  question,
  answer,
  onChange,
}: {
  question: AskedQuestion;
  answer: Answer | undefined;
  onChange: (a: Answer) => void;
}) {
  const picked = positions(answer);
  const toggle = (index: number) => {
    if (question.multiple) {
      onChange(picked.includes(index) ? picked.filter((i) => i !== index) : [...picked, index]);
    } else {
      onChange(picked.includes(index) ? [] : [index]);
    }
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {question.choices?.map((label, index) => {
        const on = picked.includes(index);
        return (
          <button
            key={index}
            type="button"
            aria-pressed={on}
            onClick={() => toggle(index)}
            className={on ? CHIP_ON : CHIP_OFF}
          >
            {label || "(empty)"}
          </button>
        );
      })}
    </div>
  );
}

function Questions({
  interrupt,
  busy,
  search,
  onResolve,
  onCancel,
}: {
  interrupt: Interrupt;
  busy: boolean;
  search: Search;
  onResolve: (payload: unknown) => void;
  onCancel: () => void;
}) {
  const stop = interrupt.metadata as Stop | undefined;
  const questions = (stop?.questions ?? []).filter((q) => q && q.name);
  const [answers, setAnswers] = useState<Record<string, Answer>>({});
  const set = (name: string, answer: Answer) => setAnswers((all) => ({ ...all, [name]: answer }));
  const submit = () => onResolve({ answers: questions.map((q) => picks(answers[q.name])) } satisfies AskAnswers);

  return (
    <div className="flex flex-col gap-4">
      {questions.map((q) => (
        <div key={q.name} className="flex flex-col gap-1.5">
          <div className="text-sm leading-5 font-medium text-wfo-ink">
            {q.title || q.question}
            {q.required === false && <span className="font-normal text-wfo-muted"> (optional)</span>}
          </div>
          {q.problem && <div className="text-xs leading-4 text-wfo-danger">{q.problem}</div>}
          {q.hint && <div className="text-xs leading-4 text-wfo-muted">{q.hint}</div>}
          {q.searchable ? (
            <Combobox
              picked={found(answers[q.name])}
              multiple={q.multiple === true}
              placeholder="Type to search"
              search={(words) => search(q.name, words)}
              onChange={(picked) => set(q.name, picked)}
            />
          ) : q.choices && q.choices.length > 0 ? (
            <Choices question={q} answer={answers[q.name]} onChange={(a) => set(q.name, a)} />
          ) : (
            <input
              value={typeof answers[q.name] === "string" ? (answers[q.name] as string) : ""}
              onChange={(e) => set(q.name, e.target.value)}
              placeholder={q.required === false ? "Leave empty to keep the default" : ""}
              className={FIELD}
            />
          )}
        </div>
      ))}
      <div className="flex items-center gap-1">
        <button type="button" className={PRIMARY} disabled={busy} onClick={submit}>
          Continue
        </button>
        <button type="button" className={GHOST} disabled={busy} onClick={onCancel}>
          Cancel form
        </button>
      </div>
    </div>
  );
}

function ApprovalCard({
  interrupt,
  busy,
  onResolve,
}: {
  interrupt: Interrupt;
  busy: boolean;
  onResolve: (payload: unknown) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {interrupt.message && <Markdown text={interrupt.message} />}
      <div className="flex items-center gap-1">
        <button
          type="button"
          className={PRIMARY}
          disabled={busy}
          onClick={() => onResolve({ approved: true } satisfies ApprovalAnswer)}
        >
          Approve and start
        </button>
        <button
          type="button"
          className={GHOST}
          disabled={busy}
          onClick={() => onResolve({ approved: false } satisfies ApprovalAnswer)}
        >
          Reject
        </button>
      </div>
      <div className="text-xs leading-4 text-wfo-muted">Nothing starts until you approve.</div>
    </div>
  );
}

/**
 * A stop of the workflow form: the questions of a page, or the start to approve, as the agent's AG-UI
 * interrupt carries it. What is answered here goes back as the run's resume entry; no model reads it.
 */
export function StopCard({
  interrupt,
  busy,
  search,
  onResolve,
  onCancel,
}: {
  interrupt: Interrupt;
  busy: boolean;
  /** The options of a searchable question that typed words fit. */
  search: Search;
  onResolve: (payload: unknown) => void;
  onCancel: () => void;
}) {
  const approval = interrupt.reason === "tool_call";
  const stop = interrupt.metadata as Stop | undefined;
  const title = approval ? "Start the workflow?" : (stop?.title ?? interrupt.message ?? "Form");
  const workflow = stop?.workflow_key;
  return (
    <div className="self-stretch overflow-hidden rounded-xl border border-wfo-line bg-wfo-bg">
      <div className="flex items-start gap-2 border-b border-wfo-line bg-wfo-accent-soft px-3.5 py-2.5">
        <FileText size={16} className="mt-0.5 shrink-0 text-wfo-accent" />
        <div className="min-w-0">
          <div className="truncate text-sm leading-5 font-semibold text-wfo-accent">{title}</div>
          {workflow && workflow !== title && (
            <div className="truncate text-xs leading-4 text-wfo-muted">Workflow {workflow}</div>
          )}
        </div>
      </div>
      <div className="px-3.5 py-3">
        {approval ? (
          <ApprovalCard interrupt={interrupt} busy={busy} onResolve={onResolve} />
        ) : (
          <Questions interrupt={interrupt} busy={busy} search={search} onResolve={onResolve} onCancel={onCancel} />
        )}
      </div>
    </div>
  );
}
