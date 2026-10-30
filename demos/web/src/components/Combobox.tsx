import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import { X } from "lucide-react";

import type { FieldOption } from "../api/types";
import { CHIP_ON, FIELD } from "./classes";

const DEBOUNCE_MS = 200;

/**
 * A long list searched instead of read: the words typed fetch the options they fit, one is picked, and the
 * pick is the answer. A list field keeps every pick as a chip.
 */
export function Combobox({
  picked,
  multiple,
  placeholder,
  search,
  onChange,
}: {
  picked: FieldOption[];
  multiple: boolean;
  placeholder: string;
  search: (words: string) => Promise<FieldOption[]>;
  onChange: (picked: FieldOption[]) => void;
}) {
  const [words, setWords] = useState("");
  const [open, setOpen] = useState(false);
  const [found, setFound] = useState<FieldOption[] | null>(null); // null: nothing asked yet
  const [active, setActive] = useState(0);
  const latest = useRef(0); // the search whose answer still counts

  useEffect(() => {
    if (!open) return;
    const asked = ++latest.current;
    const timer = setTimeout(async () => {
      try {
        const options = await search(words);
        if (latest.current === asked) setFound(options);
      } catch {
        if (latest.current === asked) setFound([]);
      }
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [words, open, search]);

  const offered = (found ?? []).filter((option) => !picked.some((pick) => pick.value === option.value));
  const pick = (option: FieldOption) => {
    onChange(multiple ? [...picked, option] : [option]);
    setWords("");
    setActive(0);
    if (!multiple) setOpen(false);
  };
  const unpick = (option: FieldOption) => onChange(picked.filter((pick) => pick.value !== option.value));
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" && offered.length) {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, offered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && open && offered[active]) {
      e.preventDefault();
      pick(offered[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "Backspace" && !words && picked.length) {
      unpick(picked[picked.length - 1]!);
    }
  };

  return (
    <div className="relative flex flex-col gap-1.5">
      {picked.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {picked.map((option) => (
            <span key={String(option.value)} className={`${CHIP_ON} gap-1 pr-1.5`}>
              {option.label}
              <button
                type="button"
                title="Remove"
                onClick={() => unpick(option)}
                className="inline-flex cursor-pointer appearance-none items-center rounded-full border-0 bg-transparent p-0.5 text-inherit hover:bg-wfo-on-accent/20"
              >
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      )}
      {(multiple || picked.length === 0) && (
        <input
          value={words}
          placeholder={placeholder}
          role="combobox"
          aria-expanded={open}
          aria-autocomplete="list"
          onChange={(e) => {
            setWords(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)} // after a click on an option lands
          onKeyDown={onKey}
          className={FIELD}
        />
      )}
      {open && (
        <ul
          role="listbox"
          className="absolute top-full right-0 left-0 z-[2] m-0 mt-1 max-h-56 list-none overflow-y-auto rounded-lg border border-wfo-line bg-wfo-bg p-1 shadow-wfo-panel"
        >
          {found === null ? (
            <li className="px-2.5 py-1.5 text-xs text-wfo-muted">Searching…</li>
          ) : offered.length === 0 ? (
            <li className="px-2.5 py-1.5 text-xs text-wfo-muted">
              {words ? "Nothing matches: try another name." : "No options."}
            </li>
          ) : (
            offered.map((option, index) => (
              <li key={String(option.value)} role="option" aria-selected={index === active}>
                <button
                  type="button"
                  onMouseDown={(e) => e.preventDefault()} // keep the input's focus until the pick lands
                  onMouseEnter={() => setActive(index)}
                  onClick={() => pick(option)}
                  className={`block w-full cursor-pointer appearance-none rounded-md border-0 px-2.5 py-1.5 text-left font-wfo text-sm text-wfo-ink ${
                    index === active ? "bg-wfo-accent-soft text-wfo-accent" : "bg-transparent"
                  }`}
                >
                  {option.label}
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
