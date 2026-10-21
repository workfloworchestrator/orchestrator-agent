import type { Artifact } from "../api/types";

/** The artifact reference a tool result carries, if it is one. */
export function artifactOf(content: string | undefined): Artifact | null {
  if (!content) return null;
  try {
    const value: unknown = JSON.parse(content);
    if (value && typeof value === "object" && ("query_id" in value || "entity_id" in value)) return value as Artifact;
  } catch {
    /* a plain tool result */
  }
  return null;
}

/** A message's text, whatever shape its content takes. */
export function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part: unknown) => (part && typeof part === "object" && "text" in part ? String(part.text) : ""))
      .join("");
  }
  return "";
}
