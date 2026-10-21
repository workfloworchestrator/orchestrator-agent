import { useCallback, useState } from "react";

import { storage } from "../lib/storage";
import { uuid } from "../lib/uuid";

/** The thread this tab is on, remembered in session storage so a reload lands on the same chat. */
export function useStoredThreadId(key: string): [string, (threadId: string) => void] {
  const [threadId, setThreadId] = useState(() => {
    const stored = storage.get(key) ?? uuid();
    storage.set(key, stored);
    return stored;
  });
  const remember = useCallback(
    (id: string) => {
      storage.set(key, id);
      setThreadId(id);
    },
    [key],
  );
  return [threadId, remember];
}
