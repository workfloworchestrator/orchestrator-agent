import { Trash2 } from "lucide-react";

import { Mark } from "./Mark";
import { ICON_BUTTON, PRIMARY } from "./classes";
import type { ChatSummary } from "../api/types";

/** When a chat was last active, as the person would say it. */
function when(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const time = date.toLocaleTimeString(undefined, { timeStyle: "short" });
  if (date.toDateString() === now.toDateString()) return time;
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return `Yesterday, ${time}`;
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: date.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

/** The person's chats as the agent keeps them: open one, forget one, or start another. */
export function ChatList({
  chats,
  loading,
  onOpen,
  onDelete,
  onNew,
}: {
  chats: ChatSummary[];
  loading: boolean;
  onOpen: (threadId: string) => void;
  onDelete: (threadId: string) => void;
  onNew: () => void;
}) {
  if (chats.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-2 text-center">
        <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-wfo-accent-soft text-wfo-accent">
          <Mark size={26} />
        </span>
        {loading ? (
          <p className="m-0 text-sm text-wfo-muted">Loading your chats…</p>
        ) : (
          <>
            <div className="text-sm font-semibold">No chats yet</div>
            <p className="m-0 max-w-[30ch] text-sm leading-5 text-wfo-muted">
              Ask about subscriptions, products and processes, or run a workflow from here.
            </p>
            <button type="button" className={PRIMARY} onClick={onNew}>
              Start a chat
            </button>
          </>
        )}
      </div>
    );
  }
  return (
    <ul className="m-0 flex list-none flex-col divide-y divide-wfo-line p-0">
      {chats.map((chat) => (
        <li
          key={chat.thread_id}
          className="group flex items-center gap-1 rounded-lg transition-colors hover:bg-wfo-surface"
        >
          <button
            type="button"
            onClick={() => onOpen(chat.thread_id)}
            className="min-w-0 flex-1 cursor-pointer appearance-none rounded-lg border-0 bg-transparent px-2 py-2.5 text-left font-wfo focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-wfo-accent"
          >
            <div className="truncate text-sm leading-5 font-medium text-wfo-ink">{chat.title}</div>
            <div className="mt-0.5 flex items-center gap-2 text-xs leading-4 text-wfo-muted">
              <span>{when(chat.updated_at)}</span>
              {chat.form_open && (
                <span className="inline-flex items-center gap-1 rounded-full bg-wfo-wait-soft px-1.5 py-px text-wfo-wait">
                  <span className="h-1.5 w-1.5 rounded-full bg-wfo-wait" />
                  Needs your answer
                </span>
              )}
            </div>
          </button>
          <button
            type="button"
            onClick={() => onDelete(chat.thread_id)}
            title="Forget this chat"
            className={`${ICON_BUTTON} mr-1 opacity-0 group-hover:opacity-100 focus-visible:opacity-100`}
          >
            <Trash2 size={16} />
          </button>
        </li>
      ))}
    </ul>
  );
}
