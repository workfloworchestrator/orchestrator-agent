# Runtime behaviour

This reference describes the agent's implementation and checks. For the roles of the connected systems, see
[How the agent fits on top of Orchestrator Core](../explanation/index.md).

## Components

pydantic-ai runs the agent. It also provides the AG-UI adapter and the MCP client, but not the other endpoints, which
are built on the SDK of their own protocol:

| Part | Built on | Documentation |
| --- | --- | --- |
| Agent loop and model calls | pydantic-ai | [pydantic-ai](https://pydantic.dev/docs/ai/) |
| AG-UI endpoint | The AG-UI adapter of pydantic-ai | [AG-UI in pydantic-ai](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/) |
| Connection to Orchestrator Core | The MCP client of pydantic-ai | [MCP client](https://pydantic.dev/docs/ai/mcp/client/) |
| A2A endpoint | The [A2A Python SDK](https://github.com/a2aproject/a2a-python) (`a2a-sdk`) | [A2A protocol](https://a2a-protocol.org/latest/) |
| MCP endpoint | `FastMCP` from the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | [MCP](https://modelcontextprotocol.io/) |
| OpenAI chat completions endpoint | The agent itself, using the types of the OpenAI SDK | [Chat API](https://platform.openai.com/docs/api-reference/chat) |

The MCP server of Orchestrator Core is a separate server, built on the [`fastmcp`](https://gofastmcp.com/) package, see
[MCP server](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/).

## Core tool compatibility and discovery

At startup the agent lists the tools of the live server and stops with an error if a tool that its code depends on
is missing. A connection failure logs a warning and skips the check. Authentication errors, HTTP errors and
malformed responses fail startup.

The check covers the fixed Core contract in `tool_names.ALL_TOOL_NAMES`. It excludes local tools such as
`start_workflow_form`; adding a constant for a custom tool does not add it to this verification list automatically.

### Find the exposed tools

Every tool is a REST route of Orchestrator Core with the tag `agent-exposed`. The API docs of your orchestrator show
them in that group, at `http://<core>/api/docs#/agent-exposed`. For each route you see the name of the tool (the
`operationId`), the description the model reads (the docstring) and the parameters.

The default set is listed in the [MCP server reference](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#usage). To add
a tool of your own, see [Extending](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending)
in the MCP server reference.

## Read tools and write tools

Most tools only read: search, aggregate, look up an entity. Orchestrator Core also exposes tools that can make
changes: `create_workflow`, `resume_workflow_process` and `abort_workflow_process`.

These write tools are part of the MCP server of Orchestrator Core, so any MCP client that connects to it directly
can call them. What the Orchestrator Agent does with them depends on the interface:

| Interface | Write tools |
| --- | --- |
| A2A, `/v1` and AG-UI | Hidden from the model by the write tool gate. These interfaces share the form-enabled agent; built-in workflow starts use the approved path described in [Workflow forms](#workflow-forms). |
| Agent `/mcp` | Not hidden. It runs a separate agent without the form-fill skill or write tool gate, so its model can call Core's write tools directly. |

The gate hides a fixed list of tool names. It does not look at the annotations of a tool, and it does not inspect what
an endpoint does. A custom route tagged `agent-exposed` is therefore visible to the model, whether or not it is marked
read-only. Conversation persistence and stored search results also write to PostgreSQL.

!!! danger "Write tools without approval"
    Where the write tools are not hidden, the model can change your inventory without the approval of a person. This
    holds for the agent `/mcp`, and for your own agent or MCP client when you connect it to Core. Use A2A,
    LibreChat or an interrupt-capable AG-UI client for the built-in approval flow, or have the calling client obtain
    confirmation. The tools carry hints for this: `abort_workflow_process` is marked destructive,
    and none of the three is marked read-only.

## Plugins and guards

A [plugin](plugins.md) is a Markdown file with a prompt and a few lines of frontmatter. It declares the
tools it owns and, optionally, how their results are shown. The built-in plugins are search, aggregate, entity,
export and workflow. Each one becomes a [capability](../explanation/index.md#capabilities) of the agent and a skill
in the A2A agent card.

Ownership of a tool drives two things. The results of the tool are mapped to the declared output type: a query result
for a table or chart, a data result for one entity, or an export. And when a plugin is set to load on demand, its
tools stay hidden together with its instructions until the model loads it. Tools owned by no plugin remain available
unless the interface's write tool gate hides them.

A few guards hold for all tools, so they are not part of a plugin. They are capabilities too, written in Python instead
of Markdown:

- The **filter path guard** requires that a call with filters or a grouping is preceded by the discovery of the valid
  filter paths. It checks for a discovery call in the available message history before a non-empty `filters` or
  `group_by` argument; it does not compare the chosen paths with the discovery result or verify its entity type.
- The **write tool gate**, on A2A, `/v1` and AG-UI (the agent with the form-fill skill), hides Core's three named
  workflow mutations from the model, so built-in workflow starts go
  through the form-fill skill. See [Read tools and write tools](#read-tools-and-write-tools).
- **History trimming** keeps a sliding window of the conversation.

Tables and charts are **rendered blocks**: the agent builds a Markdown table or a Mermaid chart from the result in
code. The A2A and chat adapters append these blocks to the answer, so the rendered rows and numbers are not retyped
by the model. AG-UI exposes the blocks as artifact metadata for the frontend to render. MCP returns the raw result
JSON alongside the model's answer. Export download metadata is delivered through AG-UI; the other interfaces
do not deliver Core's standard export URL after the plugin strips it from the model-visible payload. See
[Export delivery](endpoints.md#export-delivery).

## Workflow forms

The model selects a workflow through the workflow plugin. A form-fill skill then handles the form and approval:

1. **Route.** The model decides that a request starts a workflow and which one, from the list of workflows, and hands
   over with the key of the workflow.
2. **Walk the form.** The skill fetches the form page by page from Orchestrator Core, using the answers so far. Pages
   are dynamic, so the walk starts again from the first page every turn.
3. **Ask.** The skill asks about missing answers or fields Core rejected. It can also stop before submission for
   consent, optional fields or unresolved option choices. Questions carry the available choices and Core's errors
   where present. Consent fields require an explicit answer from the person.
4. **Confirm.** When the form is complete, the person approves the exact call that will be made. Approve starts the
   workflow and reject cancels it.
5. **Start.** The approved branch makes one start attempt with exactly the validated pages. Core validation errors
   reopen the form; other failures close it because the outcome may be unknown. This is not an exactly-once
   execution guarantee across concurrent requests or workers; see [Conversation persistence](#conversation-persistence).

No chat text is read as an answer: a person who types *yes* has not approved anything. The questions and the approval
travel through kagent's human-in-the-loop extension over A2A, LibreChat's ask-user tool, or AG-UI interrupts.
For ordinary fields, Core rejects invalid text before the agent interprets it into a value for Core to validate
again. Fields with `optionsSource` instead fetch their choices over HTTP and resolve typed words against those
choices before submission. Ambiguous or unmatched words are re-asked. Core validates every submitted page.

Core supplies each page as a self-contained JSON Schema with structured validation verdicts. The agent reads it
into `FieldSpec` records for questions and labels. Its interpreter uses an optional-field variant of that schema
through pydantic-ai's `StructuredDict`, rather than generating a Pydantic model per page. Form-option requests use
the same outbound token selection as MCP; their permitted destinations are configured by
[`WFO_CORE_API_URL` and `WFO_OPTIONS_HOSTS`](configuration.md#connection).

Approval is a structured response matching the pending request. The agent trusts its calling client to obtain
that response from a person; the protocol response itself does not prove human interaction.

The MCP endpoint runs without this skill, workflow plugin or write tool gate. A client that cannot show native
pauses on A2A, `/v1` or AG-UI gets no form, but those interfaces still hide the three workflow mutations.
The skill's internal reply text is JSON; LibreChat and AG-UI render it as Markdown alongside their structured stops.

## Conversation persistence

The agent connects directly to the WFO PostgreSQL database using `DATABASE_URI`. It records runs in `agent_runs`
and state snapshots in `graph_snapshots`, using the models supplied by Orchestrator Core. A snapshot includes
message history, query state and any open workflow form. Later turns load the latest snapshot for the conversation.

A2A, OpenAI chat completions and AG-UI replay stored message history. AG-UI takes the new turn's messages from
the frontend, replaces client-supplied state with stored state, and saves the resulting history. The MCP endpoint records
individual runs but does not load or snapshot conversation state.

| Interface | Conversation identifier |
| --- | --- |
| OpenAI chat completions | `X-Conversation-Id`; without it, each request gets a new identifier |
| A2A | `x-kagent-root-context-id`, then `x-kagent-parent-context-id`, then the A2A context id |
| AG-UI | `threadId` for stored state and history; send only the new turn's messages |
| MCP | Independent calls; no conversation continuation |

A2A and chat completions look up snapshots by conversation identifier without a user-owner check. Their clients
must isolate those identifiers between users. AG-UI binds new threads to the authenticated userinfo `sub` and checks
the owner and originating protocol when continuing a thread. Its chats API applies the same ownership boundary.
With authentication off, AG-UI threads are unowned and shared; these checks do not add isolation to A2A or chat.

A2A and chat adapters serialize turns with in-memory locks within their process. AG-UI has no corresponding turn
lock: clients should serialize requests for each thread. Database persistence does not provide cross-process
locking or workflow-start deduplication for concurrent workers or replicas.

The [chats API](endpoints.md#chats-api) lists, reads and deletes AG-UI threads, searches pending form options and
proxies exports prepared in that chat. Chat deletion removes the thread's snapshots and run records, so the
database role needs `DELETE` on `graph_snapshots` and `agent_runs` as well as the permissions used for persistence.

See [Configuration](configuration.md) for the database connection setting and
[Endpoints and protocols](endpoints.md) for conversation identifiers and client requirements.
