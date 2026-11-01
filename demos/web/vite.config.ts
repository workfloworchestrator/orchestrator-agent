import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The agent the dev server proxies to, so the page and the agent share an origin (no CORS).
const AGENT_URL = process.env.AGENT_URL ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [react()],
  // One React for the page and the popup, also when the popup is a linked checkout with its own node_modules.
  resolve: { dedupe: ["react", "react-dom"] },
  server: {
    port: 5173,
    proxy: {
      "/agui": { target: AGENT_URL, changeOrigin: true },
      "/chats": { target: AGENT_URL, changeOrigin: true },
    },
  },
});
