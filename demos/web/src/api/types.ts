/*
 * The agent's wire contract as the popup reads it. The models come from the agent's OpenAPI document
 * (openapi.ts, generated from openapi.json by `npm run contract`); what AG-UI itself carries comes from
 * @ag-ui/core.
 */

import type { Context, Interrupt, Message } from "@ag-ui/core";

import type { components } from "./openapi";

type Schemas = components["schemas"];

/** A chat as the agent lists it (``GET /chats/``). */
export type ChatSummary = Schemas["ChatSummary"];
/** The reference a query tool's result carries in place of its payload, with the chart an aggregate adds. */
export type QueryArtifact = Schemas["QueryArtifact"];
export type DataArtifact = Schemas["DataArtifact"];
export type ExportArtifact = Schemas["ExportArtifact"];
export type RenderedBlock = Schemas["RenderedBlock"];
export type ChartData = Schemas["ChartData"];
export type ChartPoint = Schemas["ChartPoint"];
/** What an ``ask_user`` interrupt carries in its metadata: a form page's questions. */
export type AskStop = Schemas["AskStop"];
export type AskedQuestion = Schemas["AskedQuestion"];
/** What a ``tool_call`` interrupt carries in its metadata: the workflow start to approve. */
export type ApprovalStop = Schemas["ApprovalStop"];
export type ToolCall = Schemas["ToolCall"];
/** The payloads that resolve the two stops. */
export type AskAnswers = Schemas["AskAnswers"];
export type ApprovalAnswer = Schemas["ApprovalAnswer"];

/** One thing the host page knows that the agent should too, as AG-UI carries it (``RunAgentInput.context``). */
export type AgentContext = Context;

/** A chat as the agent gives it back (``GET /chats/{id}``): its last messages and the form stop it waits on. */
export interface ChatDetail extends ChatSummary {
  messages: Message[];
  pending: Interrupt | null;
}

/** The reference a tool result carries in place of its payload: whichever of the agent's artifacts it is. */
export type Artifact = Partial<QueryArtifact> & Partial<DataArtifact> & Partial<ExportArtifact>;
