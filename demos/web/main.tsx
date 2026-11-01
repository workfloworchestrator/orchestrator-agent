import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@workfloworchestrator/wfo-chat/style.css";

import { App } from "./App";
import "./demo.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
