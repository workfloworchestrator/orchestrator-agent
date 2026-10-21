import type { Interrupt, Message } from "@ag-ui/core";
import { useReducer } from "react";

/** A thread as the popup shows it: what the agent stored, the run streaming now, and where things stand. */
export interface ThreadState {
  /** The messages the agent stored for the thread. */
  history: Message[];
  /** This run's messages as they stream; they join the history when the run settles. */
  run: Message[];
  /** The stops the last run ended on (a form page's questions, the start to approve), waiting for an answer. */
  interrupts: Interrupt[];
  running: boolean;
  /** The thread's history is on its way. */
  loading: boolean;
  error: string | null;
}

export type ThreadAction =
  /** Another thread is on screen: it starts empty, its history on its way when the agent has one. */
  | { type: "thread_switched"; loading: boolean }
  | { type: "history_loaded"; messages: Message[]; pending: Interrupt | null }
  | { type: "run_started" }
  | { type: "run_streamed"; messages: Message[] }
  /** The agent ended the run: on interrupts that wait for the person, or on none. */
  | { type: "run_finished"; interrupts: Interrupt[] }
  /** The run is over on this side; its messages join the history, or none when it was another thread's. */
  | { type: "run_settled"; added: Message[] | null }
  | { type: "failed"; error: string };

export const IDLE: ThreadState = { history: [], run: [], interrupts: [], running: false, loading: false, error: null };

export function threadReducer(state: ThreadState, action: ThreadAction): ThreadState {
  switch (action.type) {
    case "thread_switched":
      return { ...IDLE, running: state.running, loading: action.loading };
    case "history_loaded":
      return { ...state, history: action.messages, interrupts: action.pending ? [action.pending] : [], loading: false };
    case "run_started":
      return { ...state, running: true, error: null };
    case "run_streamed":
      return { ...state, run: action.messages };
    case "run_finished":
      return { ...state, interrupts: action.interrupts };
    case "run_settled":
      return action.added
        ? { ...state, running: false, history: [...state.history, ...action.added], run: [] }
        : { ...state, running: false };
    case "failed":
      return { ...state, error: action.error, loading: false };
  }
}

export function useThreadState() {
  return useReducer(threadReducer, IDLE);
}
