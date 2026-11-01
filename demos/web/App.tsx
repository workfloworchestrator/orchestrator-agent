import { useState } from "react";

import { WfoChat } from "@workfloworchestrator/wfo-chat";

const AGENT_URL = import.meta.env.VITE_AGENT_URL ?? "/agui/";

function remembered(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function remember(key: string, value: string): void {
  try {
    sessionStorage.setItem(key, value);
  } catch {
    /* fine */
  }
}

/** ``?theme=dark`` (or light) on the demo's address forces the popup's theme; by default it follows the system. */
function themeFromAddress(): "auto" | "light" | "dark" {
  const theme = new URLSearchParams(window.location.search).get("theme");
  return theme === "dark" || theme === "light" ? theme : "auto";
}

/** A stand-in for the page that would host the popup: it knows who is logged in. */
export function App() {
  const [token, setToken] = useState(() => remembered("demo:token"));

  return (
    <>
      <main>
        <h1>A page with the WFO agent in the corner</h1>
        <p>
          This page stands in for the WFO UI. The popup bottom right is the <code>WfoChat</code> component; it talks to
          the agent's AG-UI route at <code>{AGENT_URL}</code>. It opens on your chats; a workflow's form is filled in
          the popup, page by page, and nothing starts without your approval.
        </p>

        <label>
          Bearer token (only when the agent runs with OAuth2 on)
          <input
            value={token}
            type="password"
            onChange={(e) => {
              setToken(e.target.value);
              remember("demo:token", e.target.value);
            }}
          />
        </label>
      </main>
      <WfoChat agentUrl={AGENT_URL} token={token || undefined} theme={themeFromAddress()} defaultOpen />
    </>
  );
}
