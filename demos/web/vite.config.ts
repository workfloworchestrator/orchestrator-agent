import { fileURLToPath } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import pkg from "./package.json" with { type: "json" };

const here = (path: string) => fileURLToPath(new URL(path, import.meta.url));

// The agent the demo's dev server proxies to, so the page and the agent share an origin (no CORS).
const AGENT_URL = process.env.AGENT_URL ?? "http://127.0.0.1:8080";

export default defineConfig(({ mode }) =>
  mode === "lib"
    ? {
        // The package: one ES module and one stylesheet (the types come from tsc). React and the
        // dependencies stay the host's, installed once and deduplicated with its own.
        plugins: [react(), tailwindcss()],
        build: {
          lib: { entry: here("./src/index.ts"), formats: ["es"], fileName: "wfo-chat", cssFileName: "wfo-chat" },
          rollupOptions: {
            external: ["react", "react-dom", "react/jsx-runtime", ...Object.keys(pkg.dependencies), /^@ag-ui\//],
          },
          outDir: "dist",
        },
      }
    : {
        // The demo page: it imports the package by name, resolved to the source here.
        root: here("./demo"),
        plugins: [react(), tailwindcss()],
        resolve: {
          alias: [
            { find: "@workfloworchestrator/wfo-chat/style.css", replacement: here("./src/styles/theme.css") },
            { find: "@workfloworchestrator/wfo-chat", replacement: here("./src/index.ts") },
          ],
        },
        server: {
          port: 5173,
          proxy: {
            "/agui": { target: AGENT_URL, changeOrigin: true },
            "/chats": { target: AGENT_URL, changeOrigin: true },
          },
        },
        build: { outDir: here("./dist-site"), emptyOutDir: true },
      },
);
