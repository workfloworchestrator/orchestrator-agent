# wfo-chat

The Workflow Orchestrator agent as a chat popup in the bottom-right corner of any page. A React component
that talks to the agent over [AG-UI](https://docs.ag-ui.com): it lists the person's chats, streams the
agent's replies with the steps it takes, draws the charts the agent attaches to a result, and walks a
workflow's form page by page, with nothing started until the person approves.

## Use it

```bash
npm install @workfloworchestrator/wfo-chat
```

```tsx
import { WfoChat } from "@workfloworchestrator/wfo-chat";
import "@workfloworchestrator/wfo-chat/style.css";

<WfoChat agentUrl="/agui/" token={accessToken} context={[{ description: "The subscription on screen", value: id }]} />;
```

On a page that is not React, `mountWfoChat(props)` renders the same popup and returns the function that
removes it again.

The popup needs the agent's AG-UI route and its chats API, by default `/agui/` and `/chats` on the page's
own origin: put a proxy in front of the page (the demo's nginx config is one), or point `agentUrl` at the
agent and allow the origin there.

| Prop           | What it does                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `agentUrl`     | The agent's AG-UI route (`/agui/`).                                                            |
| `chatsUrl`     | The agent's chats API; by default next to the AG-UI route.                                     |
| `token`        | The person's bearer token; the agent forwards it to orchestrator-core.                         |
| `headers`      | Extra headers sent with every request.                                                         |
| `context`      | What the host page knows, sent as AG-UI context: the subscription on screen, the customer, ... |
| `title`        | The assistant's name in the header (`Workflow Orchestrator`).                                  |
| `suggestions`  | Opening lines offered in an empty chat; a click sends one.                                     |
| `theme`        | `auto` follows the system; `light` or `dark` pins it.                                          |
| `threadKey`    | Namespace of the current thread in session storage; one popup per key.                         |
| `historyLimit` | How many of a chat's last messages are loaded when it is opened (50).                          |
| `defaultOpen`  | Open on load.                                                                                  |

`useWfoAgent` is the hook behind the component, for a host that wants its own chat surface: the thread,
its interrupts, and `send`, `respond`, `stop`, `newChat`, `listChats`, `openChat`, `deleteChat`.

### Theming

The look is a Tailwind theme (`src/styles/theme.css`) compiled into the stylesheet: a few colour tokens,
a typeface, two shadows and two animations. A host page retunes it by overriding the variables on `:root`,
and gives it its own typeface through `--font-wfo`:

```css
:root {
  --color-wfo-accent: #1d4ed8;
  --font-wfo: "Inter", sans-serif;
}
```

The stylesheet carries no reset and no element rules, so it does not restyle the page it is dropped into.

## Develop it

```bash
npm install
AGENT_URL=http://127.0.0.1:8080 npm run dev     # the demo page on http://localhost:5173, the agent proxied
npm run check                                   # lint, format, types, build
```

The package is `src/`: `components/` the popup, its parts and their shared class strings (`classes.ts`), `hooks/` the thread over AG-UI (`useWfoAgent` composes the client binding, the chats API, the thread state and the stored thread id),
`api/` the agent's wire contract and the client for its chats API, `lib/` small helpers,
`styles/theme.css` the theme. `demo/` is a page that hosts the popup
the way the WFO UI would; it imports the package by name, resolved to the source.

### The contract with the agent

`src/api/openapi.ts` is generated from the agent's OpenAPI document, never edited. The agent is FastAPI, and
its document also names what travels inside AG-UI messages (a tool result's artifact reference with its chart
data, the metadata and answers of a form stop), so everything the popup reads from the agent beyond AG-UI
itself comes from `/openapi.json`. `src/api/types.ts` picks the schemas the popup uses out of it and adds
the AG-UI pieces (`Message`, `Interrupt`, `Context`).

```bash
npm run contract:pull    # from a running agent (AGENT_URL, default http://127.0.0.1:8085), then generate
npm run contract         # from the checked-in copy, src/api/openapi.json
```

The copy is checked in so the package builds without an agent; a test in the agent's suite fails until a
changed route or model is pulled again.

`npm run build` writes the package to `dist/` (one ES module, one stylesheet, type declarations; React is a
peer dependency and the other dependencies stay the host's). `npm run build:demo` writes the demo site to
`dist-site/`; `demo/Dockerfile` serves it with nginx, proxying `/agui/` and `/chats` to `AGENT_URL`.

## Licence

Apache 2.0, see `LICENSE`.
