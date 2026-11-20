---
description: "Add the Orchestrator Agent chat popup to the WFO UI or to your own React portal over AG-UI."
---

<!-- Remaining follow-ups:
- Publish the chat README proxy links: https://github.com/workfloworchestrator/orchestrator-agent-chat/pull/7
- Publish the UI configuration guide: https://github.com/workfloworchestrator/orchestrator-ui-library/pull/2848
- The generic portal setup below still needs its own end-to-end check.
Remove each publication item once its PR merges. -->

# How to add the agent chat to the WFO UI or your own portal

This guide shows you how to add the Orchestrator Agent as a chat popup in the bottom-right corner of the WFO UI, or of
a React portal of your own. The popup is the React component `WfoChat` from the npm package
[`@orchestrator-ui/orchestrator-agent-chat`](https://github.com/workfloworchestrator/orchestrator-agent-chat). It
talks to the agent over [AG-UI](https://docs.ag-ui.com/), shows charts and tables, offers exports as downloads, and
walks through workflow forms. The WFO UI already contains this component, so both routes need the same setup apart
from one step.

For the example stack, follow [Explore the agent inside the WFO UI](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/#part-4-explore-the-agent-inside-the-wfo-ui).
Its optional Compose overlay builds the companion UI and configures a same-origin proxy for chat, history,
form options and downloads. The steps below cover a different WFO deployment or your own React portal.

## Before you start

- The agent runs at version 0.4.0 or later. Workflow forms over AG-UI need 0.3.0, export downloads need 0.4.0. See
  [Getting started](../getting-started.md).
- You have one of these:
    - the WFO UI built on `@orchestrator-ui/orchestrator-ui-components` 11.0.0 or later; or
    - a React 18 or 19 app, built with Node.js 20 or later.
- You can configure a reverse proxy in front of the page. The agent sends no CORS headers, so the browser must reach
  the agent on the page's own origin.

## Steps

<a id="proxy-setup"></a>

### 1. Forward `/agui/` and `/chats` to the agent

The popup uses two routes of the agent: `/agui/` for the conversation and `/chats` for chat history, form options and
export downloads. By default it calls both on the page's own origin, so forward both routes from the server that
serves the page to the agent.

=== "Local development"

    Let the dev server proxy both routes. For a Vite dev server, in `vite.config.js`:

    ```js
    import { defineConfig } from "vite";

    const AGENT_URL = process.env.AGENT_URL ?? "http://127.0.0.1:8080";

    export default defineConfig({
      server: {
        proxy: {
          "/agui": { target: AGENT_URL, changeOrigin: true },
          "/chats": { target: AGENT_URL, changeOrigin: true },
        },
      },
    });
    ```

    Start the dev server with `AGENT_URL` set to the agent's address, for example
    `AGENT_URL=http://127.0.0.1:8080 npm run dev`. Other dev servers have a similar proxy setting; keep the paths
    unchanged.

=== "nginx"

    Add both routes to the reverse proxy in front of the page. For nginx:

    ```nginx
    location /agui/ {
        proxy_pass http://agent.internal:8080;
        proxy_http_version 1.1;
        proxy_buffering off;        # the agent streams its answer
        proxy_read_timeout 300s;
    }

    location /chats {
        proxy_pass http://agent.internal:8080;
    }
    ```

    Replace `http://agent.internal:8080` with the address where the proxy reaches the agent. Keep `proxy_pass`
    without a path, so the proxy passes `/agui/` and `/chats` on unchanged. Turn off response buffering for
    `/agui/`: without that, the answer appears all at once at the end instead of while the agent works.

### 2. Add the popup

=== "The WFO UI"

    The WFO UI shows the popup on every page once its configuration has `orchestratorAgentUrl`. It passes the
    signed-in person's access token to the agent, and gives the popup the colours of the current theme.

    1. Upgrade `@orchestrator-ui/orchestrator-ui-components` to 11.0.0 or later. Version 11 removes the
       Search (beta) page; see the
       [changelog](https://github.com/workfloworchestrator/orchestrator-ui-library/blob/main/CHANGELOG.md)
       for what to change in your app.
    2. In the function that builds your `OrchestratorConfig` (`getInitialOrchestratorConfig` in
       `configuration/configuration.ts` of the example UI), read an environment variable for the agent and set
       `orchestratorAgentUrl`:

        ```ts
        const {
            // ...the variables you already read
            ORCHESTRATOR_AGENT_URL,
        } = getEnvironmentVariables([
            // ...
            'ORCHESTRATOR_AGENT_URL',
        ]);

        return {
            // ...the settings you already return
            orchestratorAgentUrl: ORCHESTRATOR_AGENT_URL || undefined,
        };
        ```

        The variable name is your choice. Leaving it empty hides the popup.

    3. Set the variable to the AG-UI route as the browser reaches it, and restart the UI:

        ```shell
        ORCHESTRATOR_AGENT_URL=/agui/
        ```

=== "Your own React portal"

    1. Install the package:

        ```shell
        npm install @orchestrator-ui/orchestrator-agent-chat
        ```

    2. Render `WfoChat` once, in a component that is on every page, such as your layout:

        ```tsx
        import { WfoChat } from "@orchestrator-ui/orchestrator-agent-chat";

        export function Layout({ children, accessToken }) {
          return (
            <>
              {children}
              <WfoChat agentUrl="/agui/" token={accessToken} />
            </>
          );
        }
        ```

        The popup keeps its state in the browser and draws itself in a shadow root, so render it on the client
        only. In Next.js, load it with `dynamic(..., { ssr: false })`. It brings its own styles: there is no
        stylesheet to import, and your page's styles do not change it.

    3. Optional: pass what the page shows with `context`, so the agent can use it without the person typing it:

        ```tsx
        <WfoChat
          agentUrl="/agui/"
          token={accessToken}
          context={[{ description: "The subscription on screen", value: subscriptionId }]}
        />
        ```

    For every prop, theming with your own colours, and `mountWfoChat` for a page that does not render with React,
    see the [package README](https://github.com/workfloworchestrator/orchestrator-agent-chat#readme).

### 3. Pass the person's token

=== "The agent runs with `OAUTH2_ACTIVE=false`"

    Nothing to do. Leave `token` out. Every caller sees every chat, so do this for a local try-out only.

=== "The agent runs with `OAUTH2_ACTIVE=true`"

    The popup sends `token` as a bearer token on every request. The agent checks it, keeps each chat private to the
    person who started it, and forwards the token to Orchestrator Core, so the agent acts as that person.

    - **The WFO UI** passes the signed-in person's access token. Make sure the agent accepts tokens from the same
      identity provider as the UI and Orchestrator Core.
    - **Your own portal** passes the access token from its own sign-in. It must be issued by the identity provider
      in the agent's `OIDC_CONF_URL`, for an audience the agent and Orchestrator Core trust.

    See [How to configure authentication and authorization](secure-agent.md) and
    [Authentication and authorization](../explanation/auth.md).

## Verify it works

Open the page. A chat button appears in the bottom-right corner.

1. Open the popup and ask *How many subscriptions are there per status?*. You should get a chart.
2. Ask *Find all active subscriptions*. You should get a table.
3. Ask *Export these results to CSV.* A **Download CSV** button appears; clicking it saves the file.
4. Reload the page and open the popup again. The chat is listed under your chats.

If something does not work:

- **No chat button.** In the WFO UI, check the version of `orchestrator-ui-components` and that
  `orchestratorAgentUrl` is set; an empty value hides the popup.
- **The answer arrives all at once.** The proxy buffers `/agui/`. Turn buffering off (step 1).
- **The chat list stays empty, or the download fails with 404.** `/chats` is not forwarded to the agent (step 1).
- **CORS errors in the browser console.** `agentUrl` points at another origin. Point it at the route on the page's
  own origin and forward it to the agent (step 1).
- **401.** The token is missing, has expired, or the agent does not accept it (step 3).
- **403.** The chat belongs to another person, or was started over another protocol such as LibreChat.

## Next steps

- See what the popup can do in the use cases:
  [aggregations and charts](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/aggregate-and-charts/),
  [exports](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/export/) and
  [workflows from chat](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/workflows-from-chat/).
- [Add domain context](domain-context.md) with your identifiers and terminology.
- To build your own chat interface instead of the popup, use the `useWfoAgent` hook from the same package. See the
  [package README](https://github.com/workfloworchestrator/orchestrator-agent-chat#readme) and the
  [AG-UI endpoint reference](../reference/endpoints.md#ag-ui).

## Reference

- [Endpoints and protocols: AG-UI](../reference/endpoints.md#ag-ui)
- [Endpoints and protocols: chats API](../reference/endpoints.md#chats-api)
- [orchestrator-agent-chat on GitHub](https://github.com/workfloworchestrator/orchestrator-agent-chat)
- [AG-UI protocol documentation](https://docs.ag-ui.com/)
