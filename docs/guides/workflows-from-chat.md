# Start workflows from chat

!!! info "Goal"
    Let users start create, modify, terminate and task workflows from a chat, with the questions of the workflow form
    asked one page at a time and an approval before anything starts.

## Steps

The `workflow` [plugin](../reference/plugins.md#workflow) is built in and always on. What a client needs is a way to
show the questions and the approval to a person:

- **LibreChat.** Follow [Connect LibreChat](librechat.md). The `X-Agent-Client: librechat` header and
  `askUserQuestion: true` on the model spec are what enable forms.
- **An A2A planner.** The planner must support the kagent human-in-the-loop extension.
  See [Use the agent from an A2A planner](a2a-planner.md).

The AG-UI and MCP endpoints run without the form-fill skill, so they cannot start workflows.

### What the user goes through

1. The user asks for something that changes the inventory, for example *Create a core link between Amsterdam and
   London*. The agent lists the workflows, picks the matching one and hands over. It does not look anything up first,
   because the form offers the options itself.
2. The agent walks the form from the first page with Orchestrator Core. At each page that Orchestrator Core does not
   accept yet, the user gets one question per field: its title, whether it is required and the options, if any.
3. When the form is complete, the user sees the call that will be made: the workflow key and the values, with labels
   that say what each id stands for. A workflow that ends in a summary page has its tables shown as well.
4. **Approve** starts the workflow and **reject** cancels it. The agent answers with the id of the process.

### What the agent does not do

- It does not read a *yes* in the chat as approval. Only the approval of the pending stop counts.
- It does not judge values. Orchestrator Core validates them, and a rejected field is asked again with the message of
  Orchestrator Core. When a person describes a value in words, such as an option or a number, the agent interprets it
  once per page and shows the result at the approval.
- It does not resume or abort workflows.
- A form that is not finished is gone. When the user asks again, the form starts again.

## Verify it works

Ask *Start the terminate workflow for subscription* followed by the id of a subscription. You should be asked the
questions of the form, then asked to approve. After approval the process shows up in the process list of the
orchestrator.

## Reference

- [How it works: why workflows need approval](../architecture/index.md#why-workflows-need-approval)
- [Plugins: workflow](../reference/plugins.md#workflow)
- [Forms in Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/forms/)
