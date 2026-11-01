# The chat popup demo

A page that hosts the WFO agent's chat popup the way the WFO UI would: the `WfoChat` component from
[orchestrator-agent-chat](https://github.com/workfloworchestrator/orchestrator-agent-chat), installed from
that repository, with the agent's `/agui/` route and `/chats` API on the page's own origin.

```bash
npm install
AGENT_URL=http://127.0.0.1:8080 npm run dev     # http://localhost:5173, the agent proxied
npm run build                                   # the site in dist/; the Dockerfile serves it with nginx
```

`?theme=dark` (or `light`) on the address pins the popup's theme; the token field sets the bearer the popup
sends when the agent runs with OAuth2 on. `VITE_AGENT_URL` at build time points the popup somewhere other
than `/agui/`.

To work on the popup itself against this page, link its checkout: `npm link` there (with
`npm run build -- --watch`), then `npm link @workfloworchestrator/wfo-chat` here.
