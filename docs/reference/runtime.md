# Runtime behaviour

This reference describes the agent's implementation and checks. For the roles of the connected systems, see
[How the agent fits into WFO](../architecture/index.md).

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

### Find the exposed tools

Every tool is a REST route of Orchestrator Core with the tag `agent-exposed`. The API docs of your orchestrator show
them in that group, at `http://<core>/api/docs#/agent-exposed`. For each route you see the name of the tool (the
`operationId`), the description the model reads (the docstring) and the parameters.

The default set is listed in the [MCP server reference](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#usage). To add
a tool of your own, see [Add your own tools, services and agents](../guides/own-tools.md).

## Read tools and write tools

Most tools only read: search, aggregate, look up an entity. Orchestrator Core also exposes tools that can make
changes: `create_workflow`, `resume_workflow_process` and `abort_workflow_process`.

These write tools are part of the MCP server of Orchestrator Core, so any MCP client that connects to it directly
can call them. The Orchestrator Agent excludes them from automatic discovery. Its built-in workflow capability
starts workflows through the separate approved path described in [Workflow forms](#workflow-forms).

Every interface filters Core's tools by their MCP annotations, using `orchestrator_agent/tool_policy.py`:

| Declaration | Access through automatic discovery |
| --- | --- |
| `readOnlyHint: true` (a boolean) | Available, including newly discovered custom tools |
| `readOnlyHint: false` | Blocked |
| Missing, null or malformed declaration | Blocked |

The known workflow write tools remain blocked during automatic discovery even if incorrectly marked read-only. Core cannot supply tools
under the names reserved for local orchestration. Local form handoff and capability loading are provided by the
agent itself and need no MCP annotations. AG-UI frontend tools are excluded.

This policy trusts the configured Core server and its authors to annotate implementations accurately. It does
not inspect endpoint code or prevent a falsely marked read tool from changing business state. Adding a tagged
custom endpoint with `openapi_extra=READONLY_TOOL` makes it callable automatically once discovered; exposing it
does not add a human approval step. Conversation persistence and stored search results still write to PostgreSQL.

### Capability mutation bypass

Trusted Python capability code can return `allow_capability_mutations(toolset, tool_names={...})` from
`get_toolset()`. It exposes only the selected tool names, including local functions or Core write tools, while
the registered capability is active. Deferred capabilities must be loaded first. Unselected tools remain blocked.

The write gate requires both the explicit bypass marker and the framework's registered capability owner.
Returning the wrapper as an ordinary agent or run toolset grants no bypass. Core metadata cannot establish
capability ownership; frontend tools remain excluded. Reserved local orchestration names cannot be selected.
Declaring ownership in a Markdown plugin's `tools` field does not enable mutation access.

This bypass permits direct model invocation and does not add human confirmation. The capability author must
enforce the required approval and authorization. It does not change the built-in workflow form capability's
approval flow. See [Expose selected mutations through a capability](../guides/own-tools.md#expose-selected-mutations-through-a-capability)
for an example.

!!! danger "Connecting the write tools yourself"
    You can still connect these tools to your own agent or MCP client, at your own discretion. We do not advise it:
    the model can then change your inventory without the approval of a person. If you do, let the client ask for
    confirmation before each call. The tools carry hints for this: `abort_workflow_process` is marked destructive,
    and none of the three is marked read-only.

## Plugins and guards

A [plugin](plugins.md) is a Markdown file with a prompt and a few lines of frontmatter. It declares the
tools it owns and, optionally, how their results are shown. The built-in plugins are search, aggregate, entity,
export and workflow. Each one becomes a capability of the agent and a skill in the A2A agent card.

Ownership of a tool drives two things. The results of the tool are mapped to the declared output type: a query result
for a table or chart, a data result for one entity, or an export. And when a plugin is set to load on demand, its
tools stay hidden together with its instructions until the model loads it. Allowed tools owned by no plugin are always
available.

A few guards hold for all tools, so they are not part of a plugin:

- The **filter path guard** requires that a call with filters or a grouping is preceded by the discovery of the valid
  filter paths. Paths are specific to a database and must not be guessed.
- The **write tool gate** restricts model tools on every interface to declared reads and local orchestration,
  with an explicit selected-tool exception for active registered capabilities and no frontend exception.
  See [Read tools and write tools](#read-tools-and-write-tools).
- **History trimming** keeps a sliding window of the conversation.

Tables and charts are **rendered blocks**: the agent builds a Markdown table or a Mermaid chart from the result in code
and adds it to the answer. The model only summarises, so the numbers and rows are not retyped by a model, and they
appear on clients that only show text.

## Workflow forms

The model selects a workflow through the workflow plugin. A form-fill skill then handles the form and approval:

1. **Route.** The model decides that a request starts a workflow and which one, from the list of workflows, and hands
   over with the key of the workflow.
2. **Walk the form.** The skill fetches the form page by page from Orchestrator Core, using the answers so far. Pages
   are dynamic, so the walk starts again from the first page every turn.
3. **Ask.** At the first page that Orchestrator Core does not accept, the person gets one question per field, with
   the options as choices. Consent fields are only ever accepted by an explicit answer of the person.
4. **Confirm.** When the form is complete, the person approves the exact call that will be made. Approve starts the
   workflow and reject cancels it.
5. **Start.** The skill makes the call with exactly the validated pages. A start is sent once per approval.

No chat text is read as an answer: a person who types *yes* has not approved anything. The questions and the approval
travel as native pauses of the protocol, which is the human-in-the-loop extension of kagent over A2A, and the ask-user
tool of LibreChat. The agent never judges a value. Orchestrator Core validates every page, and the agent only
interprets what a person typed in words into a value, which Orchestrator Core validates again.

Approval is a structured response matching the pending request. The agent trusts its calling client to obtain
that response from a person; the protocol response itself does not prove human interaction.

The MCP and AG-UI endpoints run without this skill, so they have neither the workflow plugin nor the write tools.

## Conversation persistence

The agent connects directly to the WFO PostgreSQL database using `DATABASE_URI`. It records runs in `agent_runs`
and state snapshots in `graph_snapshots`, using the models supplied by Orchestrator Core. A snapshot includes
message history, query state and any open workflow form. Later turns load the latest snapshot for the conversation.

A2A and OpenAI chat completions share this conversation handling; AG-UI also uses PostgreSQL state persistence.
The MCP endpoint runs independent requests without this conversation persistence.

See [Configuration](configuration.md) for the database connection setting and
[Endpoints and protocols](endpoints.md) for conversation identifiers and client requirements.
