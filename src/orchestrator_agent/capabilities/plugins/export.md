---
id: export
description: Prepare a downloadable export of an existing query's results.
a2a_tags: [export]
examples:
  - Export the last search results as CSV
defer_loading: false
tools: [EXPORT_QUERY_TOOL]
artifact: export
---
# Exporting results

Prepare a downloadable export of an existing query's results.

## How to act
- ONLY when the user EXPLICITLY asks to export or download (words like "export", "download", "CSV",
  "spreadsheet"): prepare a downloadable export for the relevant query_id (default to the most recent
  query if none is specified). The client shows the download beside the export step: reply with exactly
  "Download has been successfully prepared." and nothing else, never a URL or a link, which does not work
  from a chat.
- Viewing/showing/getting results is NOT an export — do not prepare one for those requests.
- If a query hasn't been run yet, tell the user a search is needed first.
