import type { Interrupt, Message, ResumeEntry } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useRef } from "react";

import { chatsUrlOf, describe } from "../api/client";
import type { AgentContext, ChatSummary } from "../api/types";
import { uuid } from "../lib/uuid";
import { useAgentClient } from "./useAgentClient";
import { useChats } from "./useChats";
import { useStoredThreadId } from "./useStoredThreadId";
import { useThreadState } from "./useThreadState";

export interface UseWfoAgentOptions {
  /** The agent's AG-UI route. */
  url: string;
  /** The agent's chats API; by default the AG-UI route with ``/agui`` replaced by ``/chats``. */
  chatsUrl?: string;
  /** Sent with every request: ``Authorization`` for the person asking, anything else the deployment needs. */
  headers?: Record<string, string>;
  /** Namespace of the current thread in session storage: one per key per browser tab. */
  threadKey?: string;
  context?: AgentContext[];
  /** How many of a chat's last messages are loaded when it is opened. */
  historyLimit?: number;
}

export interface WfoAgentHandle {
  threadId: string;
  /** The thread: what the agent stored, plus this run's messages as they stream. */
  messages: Message[];
  /** The stops the last run ended on (a form page's questions, the start to approve), waiting for a response. */
  interrupts: Interrupt[];
  running: boolean;
  /** A chat is being opened or the list fetched. */
  loading: boolean;
  error: string | null;
  chats: ChatSummary[];
  send: (text: string) => Promise<void>;
  /** Answer the pending interrupts: the next run carries the entries, and the agent continues the form. */
  respond: (entries: ResumeEntry[]) => Promise<void>;
  stop: () => void;
  newChat: () => void;
  listChats: () => Promise<void>;
  openChat: (threadId: string) => Promise<void>;
  deleteChat: (threadId: string) => Promise<void>;
}

/**
 * A thread with the WFO agent over AG-UI.
 *
 * The agent keeps the thread: each run replays what it stored and records the run's messages, so a run
 * carries only its own messages (the new question, or nothing when answering a form stop). The chats API
 * lists the person's threads and gives one back with its last messages and the form stop it waits on.
 * This hook puts the pieces together: the thread on screen, its state, the client that runs it, the list.
 */
export function useWfoAgent({
  url,
  chatsUrl,
  headers,
  threadKey = "default",
  context,
  historyLimit = 50,
}: UseWfoAgentOptions): WfoAgentHandle {
  const [threadId, setThreadId] = useStoredThreadId(`wfo-chat:${threadKey}:thread`);
  const [thread, dispatch] = useThreadState();
  // The thread on screen, for what resolves after the person has moved on: a run that ends, a chat that loads.
  const onScreen = useRef(threadId);
  const client = useAgentClient({ url, threadId, headers, context, onScreen, dispatch });
  const chats = useChats({ chatsUrl: chatsUrl ?? chatsUrlOf(url), headers, historyLimit });

  // Another thread on screen at once; what the person was looking at is gone. A stored chat is empty until
  // its history arrives, a new one stays empty.
  const switchTo = useCallback(
    (id: string, stored: boolean) => {
      client.stop();
      onScreen.current = id;
      setThreadId(id);
      dispatch({ type: "thread_switched", loading: stored });
    },
    [client, dispatch, setThreadId],
  );

  const newChat = useCallback(() => switchTo(uuid(), false), [switchTo]);

  const openChat = useCallback(
    async (id: string) => {
      switchTo(id, true);
      try {
        const detail = await chats.load(id);
        if (onScreen.current !== id) return; // the person opened another chat meanwhile
        dispatch({ type: "history_loaded", messages: detail?.messages ?? [], pending: detail?.pending ?? null });
      } catch (e) {
        if (onScreen.current === id) dispatch({ type: "failed", error: describe(e) });
      }
    },
    [chats, dispatch, switchTo],
  );

  const deleteChat = useCallback(
    async (id: string) => {
      if ((await chats.remove(id)) && id === threadId) newChat();
    },
    [chats, newChat, threadId],
  );

  // The tab's thread, as the agent has it: its messages and the form stop it waits on.
  const openedRef = useRef<string | null>(null);
  useEffect(() => {
    if (openedRef.current === threadId) return;
    openedRef.current = threadId;
    void openChat(threadId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const messages = useMemo(() => [...thread.history, ...thread.run], [thread.history, thread.run]);

  return {
    threadId,
    messages,
    interrupts: thread.interrupts,
    running: thread.running,
    loading: thread.loading || chats.loading,
    error: thread.error ?? chats.error,
    chats: chats.chats,
    send: client.send,
    respond: client.respond,
    stop: client.stop,
    newChat,
    listChats: chats.list,
    openChat,
    deleteChat,
  };
}
