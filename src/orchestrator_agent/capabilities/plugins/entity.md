---
id: entity
description: Resolve a single entity by id or id-prefix.
a2a_tags: [details, lookup]
examples:
  - Show details for subscription abc123
  - Look up workflow with id prefix 4f2e
defer_loading: false
tools: [RESOLVE_ENTITY_TOOL]
artifact: data
---
# Fetching entity details

Fetch the full domain model / details for a single entity the user references.

## How to act
- When the user references an entity by id or id-prefix (and the entity type is stated or clear),
  resolve it by its id_or_prefix and entity_type — a full id or a prefix both work. On a unique
  match you get the entity; on multiple matches you get a candidate list — ask the user to pick one.
- After fetching, respond with a single short confirmation plus the key details.

IMPORTANT: Viewing, showing, getting, or "giving me" an entity means fetching its details — that is
NOT an export. Only the export capability prepares downloads.
