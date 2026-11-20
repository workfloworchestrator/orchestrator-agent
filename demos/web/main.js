// The popup on an otherwise empty page, against the agent the dev server proxies to (`npm run dev`).
import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { WfoChat } from "@orchestrator-ui/orchestrator-agent-chat";

createRoot(document.getElementById("root")).render(createElement(WfoChat, { agentUrl: "/agui/", defaultOpen: true }));
