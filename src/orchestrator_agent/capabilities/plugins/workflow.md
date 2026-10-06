---
id: workflow
description: Start a workflow (create, modify, terminate, task) by walking the caller through its input form.
a2a_tags: [workflow, form, create, modify, terminate]
examples:
  - Create a new subscription for a customer
  - Start the terminate workflow for subscription abc123
defer_loading: false
tools: [LIST_WORKFLOWS_TOOL, START_WORKFLOW_FORM_TOOL]
---
# Workflows

A request to create, modify or terminate something is a workflow start — not a search. Do not look the
product, port or subscription up first: the workflow's form lists the products and every other option
itself, and the form-fill skill asks the user for what it needs. Do exactly this:

1. List the workflows (filter by target: create / modify / terminate; for a task, list the tasks) and pick
   the one for that kind of thing.
2. Start its form with the key exactly as listed, passing the subscription id when one is known.

A dedicated form-fill skill takes over from there: it shows the user the form's questions, then the start
to approve, and starts the workflow. Its message replaces yours, so after handing off reply with one short
line. Only if two *workflows* could genuinely be meant, ask the caller which one. Never ask for form values
yourself and never fetch or fill form pages; you cannot start, resume or abort workflows. A form the user
did not finish is gone: when they ask for it again, start it again.
