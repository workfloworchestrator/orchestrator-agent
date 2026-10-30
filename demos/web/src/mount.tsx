import { createRoot } from "react-dom/client";

import { WfoChat, type WfoChatProps } from "./components/WfoChat";

/** Mount the popup on a page that is not itself React; returns the function that removes it again. */
export function mountWfoChat(props: WfoChatProps = {}, target?: HTMLElement): () => void {
  const host = target ?? document.body.appendChild(document.createElement("div"));
  const root = createRoot(host);
  root.render(<WfoChat {...props} />);
  return () => {
    root.unmount();
    if (!target) host.remove();
  };
}
