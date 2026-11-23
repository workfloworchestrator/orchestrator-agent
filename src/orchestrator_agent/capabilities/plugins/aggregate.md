---
id: aggregate
description: Count, sum, average with grouping (regular or temporal).
a2a_tags: [aggregate, analytics]
examples:
  - How many subscriptions per status?
  - Count processes grouped by type
defer_loading: false
tools: [AGGREGATE_TOOL]
artifact: query
---
# Aggregating

Count or compute statistics for the user's request.

## Steps
1. Determine the entity_type (SUBSCRIPTION, PRODUCT, WORKFLOW, or PROCESS).
2. Run the aggregation for that entity_type — count rows, or compute SUM/AVG/MIN/MAX, with any
   filters and the grouping the request implies (a regular `group_by`, or `temporal_group_by` for
   time buckets). The tool describes how to filter and how to express a breakdown; follow that.
   Set `visualization_type` only when the request asks for a chart, graph, plot or diagram (or names
   one: bar, line, pie). Otherwise leave it out: the grouped result is then shown as a table, which is
   what a caller that reads the numbers back (another agent, a report) needs.
3. Summarise the outcome in 1-2 sentences (the headline, plus any notable bucket). For a grouped or
   temporal aggregation the table — or the chart, when one was asked for — is shown to the user
   automatically; do not restate its buckets. For a bare ungrouped count, just state the number.
