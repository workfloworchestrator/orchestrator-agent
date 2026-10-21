import { type ChangeEvent, type KeyboardEvent, type RefObject, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";

import { ICON_BUTTON } from "./classes";

const MAX_HEIGHT = 120;

/** Where a question is typed: the field grows with the text, Enter sends, Shift+Enter breaks the line. */
export function Composer({
  placeholder,
  running,
  onSend,
  onStop,
  inputRef,
}: {
  placeholder: string;
  running: boolean;
  onSend: (text: string) => void;
  /** Ends the reply in progress. */
  onStop: () => void;
  /** The field itself, for whoever wants to focus it. */
  inputRef?: RefObject<HTMLTextAreaElement | null>;
}) {
  const [draft, setDraft] = useState("");
  const own = useRef<HTMLTextAreaElement>(null);
  const field = inputRef ?? own;

  const send = () => {
    setDraft("");
    if (field.current) field.current.style.height = "";
    onSend(draft);
  };
  const onChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setDraft(e.target.value);
    // The field grows with the text, up to a few lines.
    e.target.style.height = "";
    e.target.style.height = `${Math.min(e.target.scrollHeight, MAX_HEIGHT)}px`;
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && draft.trim()) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex items-end gap-2 rounded-xl border border-wfo-line bg-wfo-surface py-1.5 pr-1.5 pl-3.5 focus-within:border-wfo-accent focus-within:bg-wfo-bg">
      <textarea
        ref={field}
        rows={1}
        value={draft}
        onChange={onChange}
        onKeyDown={onKey}
        placeholder={placeholder}
        className="m-0 box-border w-full flex-1 resize-none border-0 bg-transparent p-0 py-1.5 font-wfo text-sm leading-5 text-wfo-ink outline-none placeholder:text-wfo-muted"
        style={{ maxHeight: MAX_HEIGHT }}
      />
      {running ? (
        <button
          type="button"
          onClick={onStop}
          title="Stop"
          className={`${ICON_BUTTON} bg-wfo-ink text-wfo-bg hover:bg-wfo-ink hover:text-wfo-bg`}
        >
          <Square size={14} fill="currentColor" />
        </button>
      ) : (
        <button
          type="button"
          onClick={send}
          disabled={!draft.trim()}
          title="Send"
          className={`${ICON_BUTTON} bg-wfo-accent text-wfo-on-accent hover:bg-wfo-accent-strong hover:text-wfo-on-accent disabled:bg-wfo-line disabled:text-wfo-muted disabled:opacity-100`}
        >
          <ArrowUp size={16} strokeWidth={2.5} />
        </button>
      )}
    </div>
  );
}
