/* Talking to the agent's chats API, and what to tell the person when it goes wrong. */

/** The chats API next to the AG-UI route: ``/agui/`` becomes ``/chats``, with the same origin and path prefix. */
export function chatsUrlOf(agentUrl: string): string {
  const stripped = agentUrl.replace(/\/agui\/?$/, "");
  return stripped === agentUrl ? "/chats" : `${stripped}/chats`;
}

export class HttpError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export const UNREACHABLE = "The agent could not be reached. Try again in a moment.";

/** A run the person stopped or walked away from (Stop, a new chat, another chat): not a failure. */
export function aborted(e: unknown): boolean {
  if (typeof e === "string") return /abort/i.test(e);
  return e instanceof Error && (e.name === "AbortError" || /abort/i.test(e.message));
}

/** What went wrong, in the person's terms; the agent's own words when it gave some. */
export function describe(e: unknown): string {
  const status =
    e instanceof HttpError ? e.status : Number(/HTTP (\d{3})/.exec(e instanceof Error ? e.message : "")?.[1]);
  if (status === 401) return "Sign in to use the assistant.";
  if (status === 403) return "This chat belongs to someone else.";
  if (status === 404) return "This chat no longer exists.";
  if (status >= 500) return UNREACHABLE;
  const message = e instanceof Error ? e.message : String(e);
  if (/failed to fetch|networkerror|load failed|ECONNREFUSED/i.test(message)) return UNREACHABLE;
  return message || "Something went wrong.";
}

/** A JSON request to the agent's chats API; null for a reply without a body (204). */
export async function request<T>(
  path: string,
  headers: Record<string, string> = {},
  init: RequestInit = {},
): Promise<T | null> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...headers, ...(init.headers ?? {}) },
  });
  if (!response.ok) throw new HttpError(response.status, `${response.status} ${response.statusText}`.trim());
  return response.status === 204 ? null : ((await response.json()) as T);
}
