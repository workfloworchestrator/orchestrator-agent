/** The assistant's mark: the nodes an orchestrator wires together. */
export function Mark({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M9 7h6M8.6 8.9l2.3 6.4M15.4 8.9l-2.3 6.4"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <circle cx="6.5" cy="7" r="2.6" fill="currentColor" />
      <circle cx="17.5" cy="7" r="2.6" fill="currentColor" />
      <circle cx="12" cy="17.5" r="2.6" fill="currentColor" />
    </svg>
  );
}
