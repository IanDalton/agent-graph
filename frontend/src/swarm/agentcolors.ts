/** Deterministic per-agent colour assignment for swarm-mode traces + the flow diagram.
 *
 *  Colour is keyed by the agent's id, so a given specialist (e.g. "market-researcher") is always
 *  the same colour across a turn — and the same spec dispatched twice concurrently shares a colour
 *  but lives in two separate instance bubbles. Tailwind's JIT purges dynamically-built class names
 *  (`bg-${x}-500`), so every class here is a full literal string. `hsl` is for the ReactFlow nodes
 *  (inline style, not a class). */
export interface AgentColor {
  /** Left-border accent for the bubble. */
  border: string;
  /** Subtle tinted background. */
  bg: string;
  /** Foreground text for the agent name/dot. */
  text: string;
  /** Solid swatch (the header dot, flow node). */
  dot: string;
  /** Raw HSL for inline styles (ReactFlow node border/background). */
  hsl: string;
}

const PALETTE: AgentColor[] = [
  { border: "border-l-sky-500", bg: "bg-sky-500/5", text: "text-sky-300", dot: "bg-sky-500", hsl: "hsl(199 89% 48%)" },
  { border: "border-l-violet-500", bg: "bg-violet-500/5", text: "text-violet-300", dot: "bg-violet-500", hsl: "hsl(258 90% 66%)" },
  { border: "border-l-emerald-500", bg: "bg-emerald-500/5", text: "text-emerald-300", dot: "bg-emerald-500", hsl: "hsl(160 84% 39%)" },
  { border: "border-l-amber-500", bg: "bg-amber-500/5", text: "text-amber-300", dot: "bg-amber-500", hsl: "hsl(38 92% 50%)" },
  { border: "border-l-rose-500", bg: "bg-rose-500/5", text: "text-rose-300", dot: "bg-rose-500", hsl: "hsl(347 77% 50%)" },
  { border: "border-l-cyan-500", bg: "bg-cyan-500/5", text: "text-cyan-300", dot: "bg-cyan-500", hsl: "hsl(189 94% 43%)" },
  { border: "border-l-fuchsia-500", bg: "bg-fuchsia-500/5", text: "text-fuchsia-300", dot: "bg-fuchsia-500", hsl: "hsl(292 84% 61%)" },
  { border: "border-l-lime-500", bg: "bg-lime-500/5", text: "text-lime-300", dot: "bg-lime-500", hsl: "hsl(84 81% 44%)" },
];

/** Stable hash → palette entry. Same agentId → same colour across renders and across surfaces. */
export function colorForAgent(agentId: string): AgentColor {
  let hash = 0;
  for (let i = 0; i < agentId.length; i++) {
    hash = (hash * 31 + agentId.charCodeAt(i)) >>> 0;
  }
  return PALETTE[hash % PALETTE.length];
}
