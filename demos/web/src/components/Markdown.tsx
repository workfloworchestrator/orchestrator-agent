import type { ComponentProps } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const heading = ({ node: _node, ...props }: ComponentProps<"h3"> & { node?: unknown }) => (
  <h3 className="m-0 mt-3 mb-1 text-sm font-semibold first:mt-0" {...props} />
);

/** The agent's Markdown, element by element, in the popup's own utilities: it brings no stylesheet of its own. */
const components: Components = {
  p: ({ node: _node, ...props }) => <p className="m-0 mb-2 last:mb-0" {...props} />,
  ul: ({ node: _node, ...props }) => <ul className="m-0 mb-2 list-disc pl-5 last:mb-0" {...props} />,
  ol: ({ node: _node, ...props }) => <ol className="m-0 mb-2 list-decimal pl-5 last:mb-0" {...props} />,
  h1: heading,
  h2: heading,
  h3: heading,
  a: ({ node: _node, ...props }) => (
    <a className="text-wfo-accent underline underline-offset-2" target="_blank" rel="noreferrer" {...props} />
  ),
  hr: ({ node: _node, ...props }) => <hr className="my-2 border-0 border-t border-wfo-line" {...props} />,
  code: ({ node: _node, className, ...props }) => (
    <code
      className={`rounded-[0.3rem] bg-wfo-surface px-[0.35em] py-[0.1em] font-mono text-[0.85em] ${className ?? ""}`}
      {...props}
    />
  ),
  pre: ({ node: _node, ...props }) => (
    <pre
      className="m-0 mb-2 overflow-x-auto rounded-lg bg-wfo-surface px-3 py-2 text-xs last:mb-0 [&>code]:bg-transparent [&>code]:p-0 [&>code]:text-xs"
      {...props}
    />
  ),
  table: ({ node: _node, ...props }) => (
    <table className="m-0 mb-2 block max-w-full border-collapse overflow-x-auto text-xs last:mb-0" {...props} />
  ),
  th: ({ node: _node, ...props }) => (
    <th
      className="border-0 border-b border-wfo-line px-2.5 py-1 text-left align-top text-[0.75rem] font-semibold whitespace-nowrap text-wfo-muted"
      {...props}
    />
  ),
  td: ({ node: _node, ...props }) => (
    <td className="border-0 border-b border-wfo-line px-2.5 py-1 text-left align-top whitespace-nowrap" {...props} />
  ),
  tbody: ({ node: _node, ...props }) => <tbody className="[&>tr:last-child>td]:border-b-0" {...props} />,
};

/** The agent's reply as it writes it: GitHub-flavoured Markdown, its result tables included. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-sm leading-[1.55] wrap-anywhere">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
