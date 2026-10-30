import { HttpAgent } from "@ag-ui/client";
import type { ResumeEntry } from "@ag-ui/core";
import { type Dispatch, type RefObject, useCallback, useEffect, useMemo, useRef } from "react";

import { aborted, describe } from "../api/client";
import type { AgentContext } from "../api/types";
import { uuid } from "../lib/uuid";
import type { ThreadAction } from "./useThreadState";

export interface AgentClient {
  /** Send a message on the thread; a message while a form waits ends the form first. */
  send: (text: string) => Promise<void>;
  /** Answer the pending interrupts: the next run carries the entries, and the agent continues the form. */
  respond: (entries: ResumeEntry[]) => Promise<void>;
  /** End the reply in progress. */
  stop: () => void;
}

/**
 * The AG-UI client of a thread, bound to the thread state.
 *
 * The client has state of its own (the run's messages, the pending interrupts, the stream to abort), so it
 * lives in a ref, made anew per thread, and is touched from effects and handlers only. What it streams and
 * how a run ends reach the thread state as actions.
 */
export function useAgentClient({
  url,
  threadId,
  headers,
  context,
  onScreen,
  dispatch,
}: {
  url: string;
  threadId: string;
  headers?: Record<string, string>;
  context?: AgentContext[];
  /** The thread on screen: a run that ends after the person moved on belongs to its own thread, not that one. */
  onScreen: RefObject<string>;
  dispatch: Dispatch<ThreadAction>;
}): AgentClient {
  // What the latest render knows, for the callbacks that run later.
  const headersRef = useRef(headers);
  const contextRef = useRef(context);
  useEffect(() => {
    headersRef.current = headers;
    contextRef.current = context;
  });
  const runningRef = useRef(false);
  const agentRef = useRef<HttpAgent | null>(null);

  useEffect(() => {
    const agent = new HttpAgent({ url, threadId });
    agentRef.current = agent;
    const subscription = agent.subscribe({
      onMessagesChanged: ({ messages }) => dispatch({ type: "run_streamed", messages: [...messages] }),
      onRunFinishedEvent: (params) =>
        dispatch({ type: "run_finished", interrupts: params.outcome === "interrupt" ? params.interrupts : [] }),
      onRunErrorEvent: ({ event }) => {
        // The client reports its own aborted stream as a run error too.
        if (!aborted(event.message))
          dispatch({ type: "failed", error: event.message || "The agent reported an error" });
      },
      onRunFailed: ({ error }) => {
        if (!aborted(error)) dispatch({ type: "failed", error: describe(error) });
      },
    });
    return () => {
      subscription.unsubscribe();
      agent.abortRun();
      if (agentRef.current === agent) agentRef.current = null;
    };
  }, [url, threadId, dispatch]);

  const run = useCallback(
    async (agent: HttpAgent, resume?: ResumeEntry[]) => {
      if (runningRef.current) return;
      runningRef.current = true;
      dispatch({ type: "run_started" });
      agent.headers = headersRef.current ?? {};
      try {
        await agent.runAgent({ runId: uuid(), context: contextRef.current ?? [], resume });
      } catch (e) {
        if (!aborted(e)) dispatch({ type: "failed", error: describe(e) });
      } finally {
        runningRef.current = false;
        // The run's messages join the thread; the next run carries only its own. A run that ends after the
        // person left its chat (a new chat during a reply) belongs to that chat, not to the one on screen.
        const added = [...agent.messages];
        agent.setMessages([]);
        dispatch({ type: "run_settled", added: agent.threadId === onScreen.current ? added : null });
      }
    },
    [dispatch, onScreen],
  );

  const send = useCallback(
    async (text: string) => {
      const agent = agentRef.current;
      const content = text.trim();
      if (!agent || !content || runningRef.current) return;
      // A message instead of an answer ends the form, as on every transport. The client will not start a
      // run that leaves an interrupt unaddressed, so the form is cancelled first, then the message is sent.
      if (agent.pendingInterrupts.length) {
        await run(
          agent,
          agent.pendingInterrupts.map((pending) => ({ interruptId: pending.id, status: "cancelled" as const })),
        );
        if (agent.pendingInterrupts.length) return; // the cancel did not go through: the card stays
      }
      agent.setMessages([{ id: uuid(), role: "user", content }]);
      dispatch({ type: "run_streamed", messages: [...agent.messages] });
      await run(agent);
    },
    [dispatch, run],
  );

  const respond = useCallback(
    async (entries: ResumeEntry[]) => {
      const agent = agentRef.current;
      if (!agent) return;
      agent.setMessages([]);
      await run(agent, entries);
    },
    [run],
  );

  const stop = useCallback(() => agentRef.current?.abortRun(), []);

  return useMemo(() => ({ send, respond, stop }), [send, respond, stop]);
}
