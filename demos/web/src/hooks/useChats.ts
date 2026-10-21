import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { HttpError, describe, request } from "../api/client";
import type { ChatDetail, ChatSummary, FieldOption } from "../api/types";

export interface Chats {
  chats: ChatSummary[];
  /** The list is on its way. */
  loading: boolean;
  error: string | null;
  list: () => Promise<void>;
  /** A chat with its last messages and the form stop it waits on; null for a thread the agent has not seen. */
  load: (threadId: string) => Promise<ChatDetail | null>;
  /** Forget a chat; whether the agent did. */
  remove: (threadId: string) => Promise<boolean>;
  /** The options of a searchable question of the chat's pending form stop that the words fit. */
  options: (threadId: string, field: string, words: string) => Promise<FieldOption[]>;
}

/** The person's chats as the agent keeps them, through the chats API next to the AG-UI route. */
export function useChats({
  chatsUrl,
  headers,
  historyLimit,
}: {
  chatsUrl: string;
  headers?: Record<string, string>;
  /** How many of a chat's last messages are loaded when it is opened. */
  historyLimit: number;
}): Chats {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const headersRef = useRef(headers);
  useEffect(() => {
    headersRef.current = headers;
  });

  const get = useCallback(
    <T>(path: string, init?: RequestInit) => request<T>(`${chatsUrl}${path}`, headersRef.current ?? {}, init),
    [chatsUrl],
  );

  const list = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setChats((await get<ChatSummary[]>("/?limit=50")) ?? []);
    } catch (e) {
      setError(describe(e));
    } finally {
      setLoading(false);
    }
  }, [get]);

  const load = useCallback(
    async (id: string) => {
      try {
        return await get<ChatDetail>(`/${encodeURIComponent(id)}?last=${historyLimit}`);
      } catch (e) {
        // A thread the agent has not seen yet (nothing sent on it) is simply empty.
        if (e instanceof HttpError && e.status === 404) return null;
        throw e;
      }
    },
    [get, historyLimit],
  );

  const remove = useCallback(
    async (id: string) => {
      setError(null);
      try {
        await get(`/${encodeURIComponent(id)}`, { method: "DELETE" });
        setChats((all) => all.filter((chat) => chat.thread_id !== id));
        return true;
      } catch (e) {
        setError(describe(e));
        return false;
      }
    },
    [get],
  );

  const options = useCallback(
    async (id: string, field: string, words: string) => {
      const query = new URLSearchParams({ q: words, limit: "20" });
      return (
        (await get<FieldOption[]>(`/${encodeURIComponent(id)}/options/${encodeURIComponent(field)}?${query}`)) ?? []
      );
    },
    [get],
  );

  return useMemo(
    () => ({ chats, loading, error, list, load, remove, options }),
    [chats, loading, error, list, load, remove, options],
  );
}
