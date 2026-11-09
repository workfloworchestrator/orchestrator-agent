---
id: workflow
description: Start a workflow (create, modify, terminate, task) by filling its input form with the caller.
a2a_tags: [workflow, form, create, modify, terminate]
examples:
  - Create a new subscription for a customer
  - Start the terminate workflow for subscription abc123
defer_loading: false
tools: [LIST_WORKFLOWS_TOOL, START_WORKFLOW_FORM_TOOL]
---
# Workflows

A request to create, modify or terminate something is a workflow start — not a search. Do not look the
product, port or subscription up first: the workflow's form lists its options itself. Do exactly this:

1. List the workflows (filter by target: create / modify / terminate; for a task, list the tasks) and pick
   the one for that kind of thing. Only if two *workflows* could genuinely be meant, ask which.
2. Open its form with the key exactly as listed, passing the subscription id when one is known.

The form is then filled page by page, by you, in conversation with the person: the form's instructions and
its page tools appear once it is open and say how. Use what the person already said before asking for
anything. The start itself is never yours: when the form is complete the person is shown the values and
approves the start, and only that start's own message says that something was started. You cannot start,
resume or abort workflows yourself. A form the person left is gone: when they ask for it again, open it again.
