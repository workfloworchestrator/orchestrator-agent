/** The popup's recurring controls, as class lists: one look for every button, chip and field. */

const FOCUS = "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-wfo-accent";
const BUTTON = `appearance-none inline-flex cursor-pointer items-center justify-center gap-1.5 border-0 font-wfo text-sm ${FOCUS}`;

export const PRIMARY = `${BUTTON} rounded-full bg-wfo-accent px-4 py-2 font-medium text-wfo-on-accent hover:bg-wfo-accent-strong disabled:cursor-default disabled:opacity-50`;
export const GHOST = `${BUTTON} rounded-full bg-transparent px-3 py-2 text-wfo-muted hover:bg-wfo-surface hover:text-wfo-ink disabled:cursor-default disabled:opacity-50`;
export const ICON_BUTTON = `${BUTTON} h-8 w-8 shrink-0 rounded-full bg-transparent p-0 text-wfo-muted hover:bg-wfo-surface hover:text-wfo-ink disabled:cursor-default disabled:opacity-50`;
export const CHIP = `${BUTTON} rounded-full border border-solid px-3 py-1 leading-5`;
export const CHIP_OFF = `${CHIP} border-wfo-line bg-wfo-surface text-wfo-ink hover:border-wfo-accent hover:text-wfo-accent`;
export const CHIP_ON = `${CHIP} border-wfo-accent bg-wfo-accent text-wfo-on-accent`;
export const FIELD = `box-border w-full rounded-lg border border-solid border-wfo-line bg-wfo-bg px-2.5 py-1.5 font-wfo text-sm text-wfo-ink outline-none placeholder:text-wfo-muted focus:border-wfo-accent`;
