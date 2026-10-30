// The stylesheet is built from this import (dist/wfo-chat.css); a host imports it as "@workfloworchestrator/wfo-chat/style.css".
import "./styles/theme.css";

export { WfoChat, type WfoChatProps } from "./components/WfoChat";
export { Chart } from "./components/Chart";
export { mountWfoChat } from "./mount";
export { useWfoAgent, type UseWfoAgentOptions, type WfoAgentHandle } from "./hooks/useWfoAgent";
export { chatsUrlOf } from "./api/client";
export type { AgentContext, Artifact, ChartData, ChatDetail, ChatSummary } from "./api/types";
