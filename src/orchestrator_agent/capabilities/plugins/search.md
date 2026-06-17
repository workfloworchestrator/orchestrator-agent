---
id: search
description: Find subscriptions, products, workflows, processes with lenient text/range filters.
a2a_tags: [search, query, fuzzy, semantic]
examples:
  - Find all active subscriptions
  - Search for workflows containing 'migrate'
defer_loading: false
tools: [SEARCH_TOOL]
artifact: query
---
# Searching

Find the entities the user is asking about.

## Steps
1. Determine the entity_type (SUBSCRIPTION, PRODUCT, WORKFLOW, or PROCESS) from the request.
2. Run the search for that entity_type — pass the user's text and any filters the request implies
   (concrete dimensions like status/product, and any identifiers the user gave). The tools describe
   how to build and operate on filters; follow that guidance.
3. Summarise the outcome in one sentence. A table of the results is shown to the user automatically —
   do not re-list the rows or restate them in prose.
