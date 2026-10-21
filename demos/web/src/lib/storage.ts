// Session storage remembers which thread this tab is on; the thread itself lives on the agent.
export const storage = {
  get(key: string): string | null {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string): void {
    try {
      sessionStorage.setItem(key, value);
    } catch {
      /* unavailable: the thread lives for this page only */
    }
  },
};
