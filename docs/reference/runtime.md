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
- **History trimming** keeps a sliding window of the conversation. A tool result from an earlier call is also
  bounded; see [Conversation persistence](#conversation-persistence).

Tables and charts are **rendered blocks**: the agent builds a Markdown table or a Mermaid chart from the result in
code. A grouped aggregation is a table unless the `aggregate` call set `visualization_type` (the model sets it only
when the request asks for a chart); search results are always a table. The A2A and chat adapters append blocks to the
answer, so the rendered rows and numbers are not retyped by the model. Only the non-empty blocks of the last tool
round that produced any are appended (a tool round is the tool results between two model responses, and a query that
found nothing has no block to show): the tables of earlier rounds are the model probing the schema on its way to the
real query. Identical blocks of that round are appended once. AG-UI exposes the blocks as artifact metadata for the
frontend to render. MCP returns the raw result JSON alongside the model's answer. Export download metadata is
delivered through AG-UI; the other interfaces do not deliver Core's standard export URL after the plugin strips it
from the model-visible payload. See [Export delivery](endpoints.md#export-delivery).

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

No tool result in the stored history is larger than 20,000 characters (the JSON text of its content). A larger
one, such as a grouped aggregation over every subscription, is kept as its first 20,000 characters (marker
included) with a note of its original size and that the query can be re-run. A tool result's artifact over 10,000
characters of JSON is bounded too (the client received it whole in the turn that produced it): a query artifact
keeps its reference (query, description, result count) without the table or chart it rendered, so a reopened chat
still shows which query a result came from, and any other artifact that large is dropped. This is
applied when the history is saved and when it is loaded, so threads stored before the limit existed are repaired on
their next call; each truncated result is logged with its tool name and original size. The call that produced a
result still gives the model the whole result: only the history replayed to later calls is bounded, so the model
re-runs a query when it needs rows of an earlier result again.

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

## Conversation scratch space

An agent that calls this one over A2A can have its sandboxed Python read the rows of an answer instead of having its
model retype them. This is off unless `SCRATCH_DIR` is set; see
[Conversation scratch space](configuration.md#conversation-scratch-space) for the settings and the folder layout.

When the folder exists, the agent saves the rows of the final tool round of a turn: the round whose tables
(or charts) are appended to the answer, not the probes before it. A `search` or a grouped `aggregate` result becomes one
CSV file; a scalar aggregate, an empty result and every other tool are not saved. Each saved result adds a line to the
end of the answer, naming the conversation, the file as the caller sees it (under `SCRATCH_VIRTUAL_DIR`), the number of
rows and the columns:

```text
Saved for run_python: conversation=2d2900fa-e77a-428c-a385-7f46cf16a457 file=/data/q-1.csv (152 rows; columns: customer_id, count)
```

A table with more data rows than `SCRATCH_REPLY_TABLE_MAX_ROWS` (a data row is a table row other than the header, the
separator and a `**Total**` row) is cut down when its rows are saved: the answer keeps the header, the first
`SCRATCH_REPLY_PREVIEW_ROWS` rows, the totals row if there is one, and the line
`… <n> rows in total; all rows are in /data/<file>`, followed by the file line. A table that is not cut down, and a
result without a table, still get the file line. A result whose CSV is larger than `SCRATCH_MAX_FILE_BYTES` is not
written, keeps its whole table, and is followed by `Not saved for run_python: <n> rows exceed the scratch file limit`.

Only the A2A route does this, and only for a conversation whose id is a UUID. Chat completions, AG-UI and MCP never
write files, and nothing in their answers changes. Saving is best-effort: when the folder cannot be created or a
file cannot be written (a read-only or full share), the agent logs an error with the conversation, the tool and the
reason, and the answer is the one there would be without a scratch folder: whole tables and no file line.

## Concurrency and the event loop

Every request of a process runs on one asyncio event loop: the model calls and the calls to Core wait on the network
without holding it, but work in Python that does not wait holds up every other request until it is done, including the
agent card that a readiness probe reads. Processing a large tool result is such work: the MCP client checks the
structured result against the tool's output schema, the agent renders its table or chart, and the stored history of
the conversation is converted to and from JSON. For an aggregation of 80,000 buckets these steps took a few seconds
of CPU in total, and a process that processes a larger result is silent for that long.

The agent therefore runs these steps on a small pool of worker threads (`orchestrator_agent.offload.off_loop`, four
threads, separate from the threads that serve `/health`):

| Step | Where |
| --- | --- |
| Check of a tool result against its output schema | The MCP client, for every tool call, including the form-fill skill's |
| Table and chart rendering of query results | The plugin capability that maps a tool result to its artifact |
| Conversion of the history to and from JSON | A2A, chat completions, AG-UI |
| The A2A artifact part of a result | A2A |
| The snapshot of the state, and reading it back | Conversation persistence |
| The text of a Core rejection, which embeds what it rejected | The MCP client |
| The structured data of a result | The agent `/mcp` endpoint |

A thread does not make this work faster. Python's global interpreter lock lets one thread run Python at a time, so
the work still takes the same CPU time, and two heavy requests in one process share one core. What changes is who
waits: the interpreter switches between threads every few milliseconds, so the loop gets turns while the work is
going on and the other requests and the probe are answered, where on the loop itself nothing else would run until
the work was done. One call into a library that holds the lock for its whole duration (parsing one very large JSON
document) is not interrupted, so a block of that length remains: about half a second for an aggregation of 80,000
buckets. The access to PostgreSQL (`db.session`, used by A2A, chat completions and AG-UI to load and store a
conversation) is synchronous and also runs on the loop.

For parallelism, use more processes: see [Worker processes](#worker-processes).

### Watchdog

A blocked loop cannot report itself, so a thread beside it does (`EVENT_LOOP_STALL_SECONDS`, default 5, `0` turns
it off). When the loop has not turned for that long it logs `Event loop blocked` with the stack of the loop thread,
and logs again for every further period the block lasts, so a long block produces a series of samples of where it
sits. When the loop turns again it logs `Event loop turning again` with the longest block. A call that holds the
interpreter lock is reported when it returns.

### Worker processes

uvicorn starts several processes when `WEB_CONCURRENCY` (or `--workers`) is above 1. Each has its own event loop, so a
heavy request holds up only the requests of its own process; the others accept the new connections. The default is
`1`, because these parts of the agent keep state in the memory of a process:

- **A2A tasks.** The A2A SDK keeps its tasks in memory. A request that names a task (the answer to a paused workflow
  form is a message on the same task, and so are `GetTask`, `CancelTask` and resubscribing) fails with *task does not
  exist* when it reaches another process than the one that created the task. Clients cannot be pinned to a worker
  inside a pod. The same holds for replicas of the agent behind one service without session affinity.
- **Turn locks.** A2A and chat completions serialize the turns of one conversation with a lock in the memory of the
  process. Two messages of one conversation in different processes load and store the state without it, and the last
  one wins. A workflow start is not deduplicated across processes either.

These keep their state in PostgreSQL and are safe with several processes: the AG-UI endpoint and the chats API, the
chat completions endpoint (apart from the turn lock), and the agent `/mcp` endpoint, which is stateless. A deployment
that serves only these can raise the number. Each process also costs memory (about 280 MiB when idle, and the peak of
the largest result it processes: 80,000 buckets took about 600 MiB more), a connection pool to PostgreSQL, its own
startup check of the Core tools and its own Langfuse client, so size the container and the database for the product.

See [Configuration](configuration.md) for the database connection setting and
[Endpoints and protocols](endpoints.md) for conversation identifiers and client requirements.
